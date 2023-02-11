'''实现PoB类以及PoB类中使用的辅助函数'''
import time
import copy
import numpy as np

import global_var
from consensus import Consensus
from chain import Block, BlockHead, Chain
from task import Task
from functions import hashsha256
from main import global_task_init,global_var_init

def find_task_by_id(block:Block, task_id:int) -> Task:
    '''从block或者miniblock的前一区块中找到id与task_id相符的任务\n
    会预先检查block.last的哈希是否与prehash相匹配\n
    Return:
        tasks 找到的任务，如果为None，表明没有找到 type:Task
    '''
    # if block.blockhead.prehash == block.last.calculate_blockhash(): 区块哈希链的检查交给validate
    for tasks in block.last.blockextra.task_list:
        if id(tasks) == task_id:
            return tasks
    return None

class PoB(Consensus):
    '''继承Consensus类，实现共识算法'''
    def __init__(self):
        '''初始化，目前PoB没有需要初始化的成员变量'''
        pass

    def setparam(self):
        '''没有需要设置的参数'''
        return super().setparam()

    def validate(self, lastblock:Block):
        '''验证区块链合法性\n
        param:
            lastblock 待验证的区块链的最后一个区块 type:Block
        return:
            chain_valid 如果为True表明chain有效 type:bool
        '''
        if lastblock.isGenesis: # 创世区块不需要验证
            return True

        if lastblock.last is None:
            raise ValueError(
                'Blocks other than genesis block should have a valid lastblock')

        if not lastblock or lastblock.blockextra.is_miniblock:
            raise TypeError('expect lastblock')
        blocktmp = lastblock
        prehash = blocktmp.blockhead.blockhash
        while blocktmp is not None:
            block_vali = self.validblock(blocktmp)
            blockhash = blocktmp.calculate_blockhash()
            if block_vali and int(blockhash, 16) == int(prehash, 16):
                prehash = blocktmp.blockhead.prehash
                if int(blockhash, 16) != int(blocktmp.blockhead.blockhash, 16):
                    raise Warning("blockhash not match")
                blocktmp = blocktmp.last
            else:
                return False
        return True

    def validblock(self, block:Block):
        '''验证单个区块的合法性\n
        param:
            block 待检验的区块 type:Block
        return:
            block_valid 如果为True表明block有效  type:bool
        '''
        if block.isGenesis: # 创世区块不需要验证
            return True

        if block.last is None:
            raise ValueError(
                'Blocks other than genesis block should have a valid lastblock')

        if not block or block.blockextra.is_miniblock:
            raise TypeError('expect block')
        current_task = find_task_by_id(block, block.blockextra.task_id)
        if current_task is None:
            return False

        # miniblock 哈希检查
        miniblock_list = block.blockextra.miniblock_list
        miniblock_hash = block.blockextra.miniblock_hash
        if (len(miniblock_list)) != len(miniblock_hash):
            return False
        for index, miniblock in enumerate(miniblock_list):
            if miniblock.calculate_blockhash() != miniblock_hash[index]:
                return False

        # 检验所有miniblock与block的一致性（prehash与task_id相同）
        for miniblock in miniblock_list:
            if miniblock.blockhead.prehash != block.blockhead.prehash or \
            miniblock.blockextra.task_id != block.blockextra.task_id:
                return False

        # 检查miniblock合法性并验证整合后模型在测试集上的性能
        x_test, y_test = current_task.test_set
        y_test_pred_list = []
        for miniblock in miniblock_list:
            model, miniblock_valid = self.valid_miniblock(miniblock,True)
            if not miniblock_valid:
                return False
            y_test_pred_list.append(model.predict(x_test))

        # 模型整合/评估测试集性能
        y_pred = []
        for i in range(len(x_test)):
            predictions = [x[i] for x in y_test_pred_list]
            joint_prediction = max(predictions, key=predictions.count)
            y_pred.append(joint_prediction)
        y_pred = np.array(y_pred)

        test_set_metric = current_task.metric_evaluator(y_test,y_pred) # 评估模型集成之后的性能
        # 检验测试集性能
        if test_set_metric < current_task.block_metric_requirement or \
        block.blockextra.metric != test_set_metric:
            return False

        return True

    def valid_miniblock(self, miniblock:Block, test_set_available:bool):
        '''验证miniblock的合法性\n
        param:
            miniblock 待检验的minkblock type:Block
            test_set_available 测试集可用则为True type:block
        return:(pre_block, miniblock_valid)
            miniblock_model miniblock中的模型 type:Any
            miniblock_valid 如果为True表明miniblock有效  type:bool
        '''
        if not miniblock or not miniblock.blockextra.is_miniblock:
            raise TypeError('expect miniblock')

        current_task = find_task_by_id(miniblock, miniblock.blockextra.task_id)
        if current_task is None:
            return (None,False)

        # 测试集发布后才能获得模型并评估模型性能
        if test_set_available:
            # 模型哈希以及验证集序号哈希检查
            if miniblock.blockextra.model_hash != id(miniblock.blockextra.model) or \
                miniblock.blockextra.validation_hash != hashsha256(miniblock.blockextra.validation_list):
                return (None, False)
            x_train, y_train= current_task.training_set
            x_validation_set = x_train[miniblock.blockextra.validation_list]
            y_validation_set = y_train[miniblock.blockextra.validation_list]
            validation_pred = miniblock.blockextra.model.predict(x_validation_set)
            real_metric = current_task.metric_evaluator(y_validation_set, 
                                                        validation_pred)
            if real_metric < current_task.minimum_metric or \
            real_metric != miniblock.blockextra.validation_metric:
                return (None, False)

        return (miniblock.blockextra.model, True)

    def train(self, lastblock: Block, miner, is_adversary):
        """
        从最新区块获得任务,训练产生弱模型并返回miniblock\n
        param:
            lastblock 最新区块 type:Block
            miner 当前矿工 type:int
        return (new_miniblock, mining_success)
            new_miniblock 新的miniblock type:None(未产生有效miniblock)/Block
            training_success 训练成功标识 type:Bool
        """
        # TODO 如果task_list中不止一个任务，可以改为获取费用最高的任务
        current_task = lastblock.blockextra.task_list[0]
        model = current_task.model_constructor()
        metric_evaluator = current_task.metric_evaluator
        x_train, y_train = current_task.training_set
        bag_scale = current_task.bag_scale
        # Bootstrap
        indexes = np.random.randint(0, len(x_train), int(len(x_train)*bag_scale))
        bag = x_train[indexes]
        target = y_train[indexes]
        # 训练
        model.fit(bag, target)
        # 生成验证集
        out_of_bag_sample_indexes = [x for x in range(len(x_train)) if x not in indexes]
        validation_list = out_of_bag_sample_indexes # TODO 从袋外样本中随机选择一部分作为验证集，而不是所有袋外样本构成验证集
        x_validation_set = x_train[validation_list]
        y_validation_set = y_train[validation_list]
        y_validation_predict = model.predict(x_validation_set)
        validation_hash = hashsha256(validation_list)
        validation_metric = metric_evaluator(y_validation_set, y_validation_predict)

        # 判断性能指标是否符合最低性能要求
        if validation_metric < current_task.minimum_metric:
            return (None, False)

        blockextra = Block.BlockExtra(id(current_task), lastblock.blockextra.task_list, 
                                      None, None, None, True, 
                                      model, id(model), validation_hash, 
                                      validation_list, validation_metric)
        # miniblock中继承上一个区块中的task_list
        blockhead = BlockHead(lastblock.blockhead.blockhash, None, 
                              time.time_ns(), None, None, lastblock.blockhead.height+1, miner) 
        new_miniblock = Block(None, blockhead, None, is_adversary, blockextra, 
                              False, global_var.get_blocksize())
        new_miniblock.blockhead.blockhash = new_miniblock.calculate_blockhash()
        new_miniblock.last = lastblock
        return (new_miniblock, True)

    def mining_consensus(self, miniblock_list, miner, content, is_adversary):
        """
        整合模型并构建区块\n
        param:
            miniblock_list miniblock列表 type:list
            miner 当前矿工ID type:int
            content 写入区块的内容 type:any
        return: (new_block, pow_success)
            new_block 新区块 type:None(未挖出新块)/Block
            mining_success 挖矿成功标识 type:Bool
        """
        # TODO miniblock_list中的miniblock具有不同的prehash或者不同task_id的情况，需要在Miner.Mining进行处置
        # 根据ID找到任务
        current_task = find_task_by_id(miniblock_list[0], miniblock_list[0].blockextra.task_id)
        if current_task is None:
            return (None,False)
        x_test,y_test = current_task.test_set
        metric_evaluator = current_task.metric_evaluator
        miniblock_num = current_task.miniblock_num
        minimum_metric = current_task.block_metric_requirement

        # 抽取数量为miniblock_num的miniblock，用其中的模型在测试集上预测
        if len(miniblock_list) > miniblock_num:
            miniblock_sample_index = np.random.randint(0,len(miniblock_list),miniblock_num)
            miniblock_sampled_list = miniblock_list[miniblock_sample_index]
        else:
            miniblock_sampled_list = miniblock_list
        miniblock_preict = []
        for miniblock in miniblock_sampled_list:
            miniblock_preict.append(miniblock.blockextra.model.predict(x_test))

        # 模型整合/评估测试集性能
        y_pred = []
        for i in range(len(x_test)):
            predictions = [x[i] for x in miniblock_preict]
            joint_prediction = max(predictions, key=predictions.count)
            y_pred.append(joint_prediction)
        y_pred = np.array(y_pred)

        metric = metric_evaluator(y_test,y_pred) # 评估模型集成之后的性能
        # 判断性能指标是否符合最低性能要求
        if metric < minimum_metric:
            return (None, False)

        miniblock_hash = [miniblock.calculate_blockhash() for miniblock in miniblock_sampled_list]
        blockhead = BlockHead(miniblock_list[0].blockhead.prehash, None, time.time_ns(),
                              None, None, miniblock_list[0].blockhead.height, miner)
        # TODO 对下一个区块中的task_list进行处理，目前使用前一区块的任务组成当前区块的task_list
        blockextra = Block.BlockExtra(id(current_task),
                                      copy.copy(miniblock_list[0].blockextra.task_list),
                                      miniblock_hash, miniblock_sampled_list, metric,
                                      False, None, None, None, None, None)
        new_block = Block(''.join(['B',str(global_var.get_block_number())]),
                          blockhead, content, is_adversary, blockextra,
                          False, global_var.get_blocksize())
        new_block.blockhead.blockhash = new_block.calculate_blockhash() # 更新区块哈希信息
        return (new_block, True)

# 对PoB类中的函数进行简单验证
if __name__ == "__main__":
    N = 5 # 矿工数量，不能小于miniblock的数量
    global_var_init(N,5,16)
    global_task_init()
    print('Initialization finished')
    test_chain_1 = Chain()
    test_chain_2 = Chain()
    consensus = PoB()
    # 生成miniblock
    miniblock_list_test = []
    for miner_id in range(N):
        new_miniblock_test,training_success = consensus.train(test_chain_1.lastblock, miner_id, False)
        if not training_success:
            raise Exception("Training failed for some reason")
        print('Miniblock from miner', miner_id, 'ready')
        miniblock_list_test.append(new_miniblock_test)
    # 出块
    new_block_test, mining_success = consensus.mining_consensus(miniblock_list_test, 0, 12, False)
    if mining_success:
        print('Block successfully generated')
    else:
        raise Exception('Failed to mine a new block')
    # 将新块添加到链中
    test_chain_1.AddBlock(new_block_test)
    # 尝试验证该区块、添加到第二条测试链并输出结果
    if consensus.validate(new_block_test):
        test_chain_2.AddChain(new_block_test)
        # 找到最长链（假设采用最长链机制）
        depth_2 = test_chain_2.lastblock.BlockHeight()
        depth_new_block = new_block_test.BlockHeight()
        if depth_2 < depth_new_block:
            test_chain_2.lastblock = new_block_test

        test_chain_2.ShowLChain()
        test_chain_2.ShowStructure1()
        print('Block metric', test_chain_2.lastblock.blockextra.metric)
        miniblock_metrics = [miniblock.blockextra.validation_metric for miniblock \
                             in test_chain_2.lastblock.blockextra.miniblock_list]
        print('Miniblock validation metric', miniblock_metrics)

    else:
        print("Validation of a new block failed")
