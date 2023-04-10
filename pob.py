'''实现PoB类以及PoB类中使用的辅助函数'''
import time
import copy
import random
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
    return:
        tasks 找到的任务，如果为None，表明没有找到 type:Task
    '''
    # if block.blockhead.prehash == block.last.calculate_blockhash(): 区块哈希链的检查交给validate
    task = block.last.blockextra.task_queue[0]
    if id(task) == task_id:
        return task
    return None

class PoB(Consensus):
    '''继承Consensus类，实现共识算法'''
    def __init__(self):
        '''初始化，目前PoB没有需要初始化的成员变量'''
        pass

    def setparam(self):
        '''没有需要设置的参数'''
        return super().setparam()
    
    def validate_evaluate_miniblock(self, miniblock_list, current_task:Task,
                                    dataset_type:Task.DatasetType):
        '''检查miniblock合法性并验证整合后模型在指定数据集上的性能\n
        param:
            miniblock_list 携带模型信息的miniblock列表 type:list
            current_task 当前任务 type:Task
            dataset_type 测试集/验证集 type:Task.DatasetType
        return:
            metric 指定数据集上的性能指标'''
        if dataset_type is Task.DatasetType.TEST_SET:
            x, y = current_task.test_set
        elif dataset_type is Task.DatasetType.VALIDATION_SET:
            x, y = current_task.validation_set
        y_pred_list = []
        for miniblock in miniblock_list:
            model, miniblock_valid = self.valid_miniblock(miniblock)
            if not miniblock_valid:
                return None
            y_pred_list.append(model.predict(x))

        # 模型整合/评估性能
        y_pred = []
        for i in range(len(x)):
            predictions = [prediction[i] for prediction in y_pred_list]
            joint_prediction = max(predictions, key=predictions.count)
            y_pred.append(joint_prediction)
        y_pred = np.array(y_pred)

        return current_task.metric_evaluator(y,y_pred) # 评估模型集成之后的性能

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

        if not lastblock or \
            lastblock.blockextra.blocktype is lastblock.BlockType.MINIBLOCK:
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

        if not block or block.blockextra.blocktype is block.BlockType.MINIBLOCK:
            raise TypeError('expect block')
        current_task = find_task_by_id(block, block.blockextra.task_id)
        if current_task is None:
            return False

        # 任务队列检查
        if len(block.blockextra.task_queue) > global_var.get_task_queue_length():
            raise Warning('Task queue in some block too long')

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
        test_set_metric = self.validate_evaluate_miniblock(miniblock_list,
                                                           current_task,
                                                           Task.DatasetType.TEST_SET)
        # 检验测试集性能
        if test_set_metric < current_task.block_metric_requirement or \
        block.blockextra.metric != test_set_metric:
            return False

        return True

    def valid_miniblock(self, miniblock:Block):
        '''验证miniblock的合法性\n
        param:
            miniblock 待检验的minkblock type:Block
        return:(pre_block, miniblock_valid)
            miniblock_model miniblock中的模型 type:Any
            miniblock_valid 如果为True表明miniblock有效  type:bool
        '''
        if not miniblock or \
            miniblock.blockextra.blocktype is not miniblock.BlockType.MINIBLOCK:
            raise TypeError('expect miniblock')

        current_task = find_task_by_id(miniblock, miniblock.blockextra.task_id)
        if current_task is None:
            return (None,False)

        return (miniblock.blockextra.model, True)

    def train(self, lastblock: Block, miner, validate_metric, is_adversary):
        """
        从最新区块获得任务,训练产生弱模型并返回miniblock\n
        param:
            lastblock 最新区块 type:Block
            miner 当前矿工 type:int
        return (new_miniblock, mining_success)
            new_miniblock 新的miniblock type:None(未产生有效miniblock)/Block
            training_success 训练成功标识 type:Bool
        """
        current_task = lastblock.blockextra.task_queue[0]
        model = current_task.model_constructor()
        x_train, y_train = current_task.training_set
        bag_scale = current_task.bag_scale
        # Bootstrap
        indexes = np.random.randint(0, len(x_train), int(len(x_train)*bag_scale))
        bag = x_train[indexes]
        target = y_train[indexes]
        # 训练
        model.fit(bag, target)
        # 构造miniblock
        blockextra = Block.BlockExtra(id(current_task), None, None, None, None,
                                      None, lastblock.BlockType.MINIBLOCK,
                                      model, id(model), validate_metric)
        # miniblock中继承上一个区块中的task_list
        blockhead = BlockHead(lastblock.blockhead.blockhash, None,
                              time.time_ns(), None, None, lastblock.blockhead.height+1, miner)
        new_miniblock = Block(global_var.get_miniblock_name(lastblock.name), blockhead, None,
                              is_adversary, blockextra, False, global_var.get_miniblock_size())
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
        # miniblock_list中的miniblock具有不同的prehash或者不同task_id的情况，需要在Miner.Mining进行处置
        # 根据ID找到任务
        current_task = find_task_by_id(miniblock_list[0], miniblock_list[0].blockextra.task_id)
        if current_task is None:
            return (None,False)
        minimum_metric = current_task.block_metric_requirement
        
        # 评估模型集成之后的性能
        metric = self.validate_evaluate_miniblock(miniblock_list,
                                                  current_task,
                                                  Task.DatasetType.TEST_SET)
        # 判断性能指标是否符合最低性能要求
        if metric < minimum_metric:
            return (None, False)

        miniblock_hash = [miniblock.calculate_blockhash() for miniblock in miniblock_list]
        blockhead = BlockHead(miniblock_list[0].blockhead.prehash, None, time.time_ns(),
                              None, None, miniblock_list[0].blockhead.height, miner)
        current_task_queue = miniblock_list[0].last.blockextra.task_queue
        current_task_list = miniblock_list[0].last.blockextra.task_list
        new_task_queue = current_task_queue[1:]
        new_task_queue.append(current_task_list[0])

        blockextra = Block.BlockExtra(id(current_task),
                                      copy.copy(current_task_list), new_task_queue,
                                      miniblock_hash, miniblock_list, metric,
                                      Block.BlockType.BLOCK)
        new_block = Block(''.join(['B',str(global_var.get_block_number())]),
                          blockhead, content, is_adversary, blockextra,
                          False, global_var.get_blocksize())
        new_block.blockhead.blockhash = new_block.calculate_blockhash() # 更新区块哈希信息
        return (new_block, True)

# 对PoB类中的函数进行简单验证
if __name__ == "__main__":
    N = 3 # 矿工数量，不能小于miniblock的数量
    global_var_init(N,5,16,2,None)
    global_var.set_block_metric_requirement(0.85)
    global_task_init()
    print('Initialization finished')
    test_chain_1 = Chain()
    test_chain_2 = Chain()
    consensus = PoB()
    # 生成miniblock
    miniblock_list_test = []
    for miner_id in range(N):
        new_miniblock_test,training_success = consensus.train(test_chain_1.lastblock,
                                                              0,miner_id, False)
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
        print('Test set metric', consensus.validate_evaluate_miniblock(
            test_chain_2.lastblock.blockextra.miniblock_list,
            test_chain_2.lastblock.last.blockextra.task_queue[0],
            Task.DatasetType.TEST_SET))
        print('Validate set metric', consensus.validate_evaluate_miniblock(
            test_chain_2.lastblock.blockextra.miniblock_list,
            test_chain_2.lastblock.last.blockextra.task_queue[0],
            Task.DatasetType.VALIDATION_SET))

    else:
        print("Validation of a new block failed")
