'''实现PoB类以及PoB类中使用的辅助函数'''
import time
import copy
import random
import numpy as np
import logging
from typing import Tuple, List
from numpy import ndarray

import global_var
from consensus import Consensus
from chain import Block, BlockHead, Chain, BlockList
from task import Task
from functions import hashsha256
from main import global_task_init,global_var_init
from arcing import arc, bagging, aggregate_and_predict_proba

ensemble_method = arc

logger = logging.getLogger(__name__)

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
        '''初始化'''
        self.target = '0'
        self.ctr=0 #计数器
        # 通过evaluation_cache缓存prdict_array，以Miniblock的哈希值为键，将测试集与验证集上模型的预测结果装入列表作为值
        self.evaluation_cache:dict[str,list[ndarray]] = {}
        self.current_height = 0 # 保存当前高度
        self.ensemble_block_validation_cache:list[str] = [] # 缓存已经验证通过的ensemble_block的哈希
        self.ensemble_block_metric_cache:dict[str,float] = {}

    def setparam(self, target):
        '''设置Key Block生成的难度值'''
        self.target = target
        return super().setparam()

    def validate_evaluate_miniblock(self, miniblock_list, current_task:Task,
                                    dataset_type:Task.DatasetType):
        '''检查miniblock合法性并验证整合后模型在指定数据集上的性能\n
        param:
            miniblock_list 携带模型信息的miniblock列表 type:list
            current_task 当前任务 type:Task
            dataset_type 测试集/验证集 type:Task.DatasetType
        return:
            metric 指定数据集上的性能指标
        '''
        if self.current_height < miniblock_list[0].blockhead.height:
            self.evaluation_cache = {} # 清空缓存
            self.current_height = miniblock_list[0].blockhead.height
        # elif self.current_height > miniblock_list[0].blockhead.height:
            # logger.info("cache miss for miniblock %s",miniblock_list[0].name)

        if dataset_type is Task.DatasetType.TEST_SET:
            x, y = current_task.test_set
            position = 0
        elif dataset_type is Task.DatasetType.VALIDATION_SET:
            x, y = current_task.validation_set
            position = 1
        else:
            raise ValueError("evaluation on training set not allowed")
        
        y_pred_list = []
        for miniblock in miniblock_list:
            model, miniblock_valid = self.valid_miniblock(miniblock)
            if not miniblock_valid:
                return None
            miniblock_hash = miniblock.blockhead.blockhash
            self.evaluation_cache.setdefault(miniblock_hash, [None,None])
            cache = self.evaluation_cache[miniblock_hash]
            if cache[position] is not None:
                y_pred_list.append(cache[position])
            else:
                predict_array = aggregate_and_predict_proba(model, x)
                y_pred_list.append(predict_array)
                cache[position] = predict_array

        # Alternative: majority voting instead of probablistic combining
        # 模型的概率合并
        classes = model[0].classes_
        y_pred_merge = np.concatenate(y_pred_list, axis=2)
        y_pred_proba = np.mean(y_pred_merge, axis=2)
        y_pred = classes.take(np.argmax(y_pred_proba, axis=1), axis=0)

        return current_task.metric_evaluator(y,y_pred) # 评估模型集成之后的性能

    def valid_chain(self, lastblock:Block):
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
            lastblock.blockextra.blocktype is not lastblock.BlockType.KEY_BLOCK:
            raise TypeError('expect key block as lastblock')
        blocktmp = lastblock
        prehash = blocktmp.blockhead.blockhash
        while blocktmp is not None:
            block_vali = self.validate_key_block(blocktmp)
            blockhash = blocktmp.calculate_blockhash()
            if block_vali and int(blockhash, 16) == int(prehash, 16):
                prehash = blocktmp.blockhead.prehash
                if int(blockhash, 16) != int(blocktmp.blockhead.blockhash, 16):
                    raise Warning("blockhash not match")
                blocktmp = blocktmp.last
            else:
                return False
        return True

    def valid_partial(self, lastblock: Block, 
                      local_chain: Chain) -> Tuple[List[Block], Block]:
        '''验证某条链上不在本地链中的区块
        param:
            lastblock 要验证的链的最后一个区块 type:Block
            local_chain 本地区块链 tyep:Chain
        return:
            copylist 需要拷贝的区块list type:List[Block]
            insert_point 新链的插入点 type:Block
        '''
        receive_tmp = lastblock
        if not receive_tmp:  # 接受的链为空，直接返回
            return (None, None)
        copylist = []
        local_tmp = local_chain.search(receive_tmp)
        ss = receive_tmp.calculate_blockhash()
        while receive_tmp and not local_tmp:
            block_vali = self.validate_key_block(receive_tmp)
            hash = receive_tmp.calculate_blockhash()
            if block_vali and int(hash, 16) == int(ss, 16):
                ss = receive_tmp.blockhead.prehash
                copylist.append(receive_tmp)
                receive_tmp = receive_tmp.last
                local_tmp = local_chain.search(receive_tmp)
            else:
                return (None, None)
        if int(receive_tmp.calculate_blockhash(), 16) == int(ss, 16):
            return (copylist, local_tmp)
        else:
            return (None, None)
    
    def validate_key_block(self, block:Block):
        '''验证Key Block的合法性\n
        param:
            block 待检验的Key Block type:Block
        return:
            block_valid 如果为True表明block有效  type:bool
        '''
        if block.isGenesis: # 创世区块不需要验证
            return True
        if block.last is None:
            raise ValueError(
                'Blocks other than genesis block should have a valid lastblock')
        if not block or block.blockextra.blocktype is not block.BlockType.KEY_BLOCK:
            raise TypeError('expect key block')
        
        # PoW验证
        target = block.blockhead.target
        blockhash = block.calculate_blockhash()
        if int(blockhash, 16) >= int(target, 16):
            return False

        current_task = find_task_by_id(block, block.blockextra.task_id)
        if current_task is None:
            return False
        # 任务队列检查
        if len(block.blockextra.task_queue) > global_var.get_task_queue_length():
            raise Warning('Task queue in some block too long')

        validate_list = block.blockextra.validate_list
        if len(validate_list) != len(block.blockextra.ensemble_block_list):
            return False

        for ensemble_block in block.blockextra.ensemble_block_list:
            # 检验所有Ensemble Block与Key Block的一致性（prehash与task_id相同）
            if ensemble_block.blockhead.prehash != block.blockhead.prehash or \
                ensemble_block.blockextra.task_id != block.blockextra.task_id:
                return False
            # 检查Ensemble Block的哈希以及validate list中各项
            blockhash = ensemble_block.calculate_blockhash()
            if not self.valid_block(ensemble_block) or \
                blockhash not in block.blockextra.validate_list:
                return False
            metric = self.ensemble_block_metric_cache.get(blockhash,None)
            if metric is None:
                metric = self.validate_evaluate_miniblock(
                ensemble_block.blockextra.miniblock_list, current_task,
                Task.DatasetType.VALIDATION_SET)
                self.ensemble_block_metric_cache[blockhash] = metric
            if metric is None or validate_list[blockhash] != metric:
                return False
        # 验证metric是否对应性能最好的Ensemble Block
        if max(validate_list.values()) != block.blockextra.metric:
            return False
        return True

    def valid_block(self, block:Block):
        '''验证Ensemble Block的合法性\n
        param:
            block 待检验的Ensemble Block type:Block
        return:
            block_valid 如果为True表明block有效  type:bool
        '''
        if block.last is None:
            raise ValueError(
                'Blocks other than genesis block should have a valid lastblock')

        if not block or block.blockextra.blocktype is not block.BlockType.BLOCK:
            raise TypeError('expect ensemble block')
        
        if block.blockhead.blockhash in self.ensemble_block_validation_cache:
            return True
        
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
        test_set_metric = self.validate_evaluate_miniblock(miniblock_list,
                                                           current_task,
                                                           Task.DatasetType.TEST_SET)
        # 检验测试集性能
        if test_set_metric < current_task.block_metric_requirement or \
        block.blockextra.metric != test_set_metric:
            return False
        # 确定ensemble block有效之后再将其放入ensemble_block_validation_cache
        self.ensemble_block_validation_cache.append(block.blockhead.blockhash)
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
        current_task = lastblock.blockextra.task_queue[0]
        x_train, y_train = current_task.training_set
        # generate a sequence of models with arc
        model_seq, _ = ensemble_method(global_var.get_arcing_round(), x_train, y_train,
                       current_task.model_constructor)
        model_hash = hashsha256([id(model) for model in model_seq]) # hash concatenation of model id
        blockextra = Block.BlockExtra(id(current_task), None, None, None, None,
                                      None, lastblock.BlockType.MINIBLOCK,
                                      model_seq, model_hash)
        # miniblock中继承上一个区块中的task_list
        blockhead = BlockHead(lastblock.blockhead.blockhash, None,
                              time.time_ns(), None, None, lastblock.blockhead.height+1, miner)
        new_miniblock = Block(global_var.get_miniblock_name(lastblock.name), blockhead, None,
                              is_adversary, blockextra, False, global_var.get_miniblock_size())
        new_miniblock.blockhead.blockhash = new_miniblock.calculate_blockhash()
        new_miniblock.last = lastblock
        return (new_miniblock, True)

    def ensemble(self, miniblock_list:BlockList, miner, is_adversary):
        """
        整合模型并构建Ensemble Block\n
        param:
            miniblock_list miniblock列表 type:list
            miner 当前矿工ID type:int
            content 写入区块的内容 type:any
        return: (new_block, pow_success)
            new_block 新Ensemble Block type:None(未挖出新块)/Block
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

        blockextra = Block.BlockExtra(id(current_task),
                                      None, None,
                                      miniblock_hash, miniblock_list.copy(),
                                      metric, Block.BlockType.BLOCK)
        new_block = Block(''.join(['B',str(global_var.get_block_number())]),
                          blockhead, None, is_adversary, blockextra,
                          False, global_var.get_miniblock_size()) # Ensemble Block与miniblock具有相同大小
        new_block.blockhead.blockhash = new_block.calculate_blockhash() # 更新区块哈希信息
        new_block.last = miniblock_list[0].last
        return (new_block, True)

    def mining_consensus(self, ensemble_block_list:BlockList, miner, content, is_adversary, q):
        """
        产生Key Block\n
        param:
            ensemble_block_list Ensemble Block列表 type:list
            miner 当前矿工ID type:int
            content 写入区块的内容 type:any
            q 每个矿工每轮次可计算哈希的次数
        return: (new_block, pow_success)
            new_block 新Ensemble Block type:None(未挖出新块)/Block
            mining_success 挖矿成功标识 type:Bool
        """
        if len(ensemble_block_list) == 0:
            return (None, False)

        # 根据ID找到任务
        current_task = find_task_by_id(ensemble_block_list[0],
                                       ensemble_block_list[0].blockextra.task_id)
        if current_task is None:
            return (None,False)

        validate_list={}
        # 逐一评估在验证集上性能
        for ensemble_block in ensemble_block_list:
            blockhash = ensemble_block.blockhead.blockhash
            if blockhash in validate_list:
                raise Warning('Duplicate items in ensemble block list')

            metric = self.ensemble_block_metric_cache.get(blockhash,None)
            if metric is None:
                metric = self.validate_evaluate_miniblock(
                                            ensemble_block.blockextra.miniblock_list,
                                            current_task, Task.DatasetType.VALIDATION_SET)
                self.ensemble_block_metric_cache[blockhash] = metric

            if metric is not None:
                validate_list[blockhash] = metric
            else:
                return (None, False)

        blockhead = BlockHead(ensemble_block_list[0].blockhead.prehash, None, time.time_ns(),
                              self.target, None, ensemble_block_list[0].blockhead.height, miner)
        current_task_queue = ensemble_block_list[0].last.blockextra.task_queue
        current_task_list = ensemble_block_list[0].last.blockextra.task_list
        new_task_queue = current_task_queue[1:]
        new_task_queue.append(current_task_list[0])
        optimal_metric = max(validate_list.values()) # 由于指标采用准确率，取最高准确率

        blockextra = Block.BlockExtra(id(current_task),
                                      copy.copy(current_task_list), new_task_queue,
                                      None, None, optimal_metric, Block.BlockType.KEY_BLOCK,
                                      None, None, validate_list, ensemble_block_list.copy())
        new_block = Block(None, blockhead, content, is_adversary, blockextra,
                          False, global_var.get_blocksize())
        for i in range(q):
            new_block.blockhead.nonce = self.ctr
            current_hash = new_block.calculate_blockhash() # 计算区块哈希信息
            if int(current_hash,16) < int(self.target,16):
                # 找到有效的PoW
                new_block.name = 'K'+str(global_var.get_key_block_number())
                new_block.blockhead.blockhash = current_hash
                self.ctr = 0
                return (new_block, True)
            self.ctr += 1
        return (None, False)

# 对PoB类中的函数进行简单验证
if __name__ == "__main__":
    N = 3 # 矿工数量，不能小于miniblock的数量
    global_var_init(N,5,16,2)
    global_var.set_block_metric_requirement(0.85)
    global_task_init()
    print('Initialization finished')
    test_chain_1 = Chain()
    test_chain_2 = Chain()
    test_chain_1.create_genesis_block()
    test_chain_2.create_genesis_block()
    consensus = PoB()
    consensus.setparam('7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF')
    # 生成miniblock
    miniblock_list_test = []
    import time
    trainging_start = time.time()
    for miner_id in range(N):
        new_miniblock_test,training_success = consensus.train(test_chain_1.lastblock,
                                                              miner_id, False)
        if not training_success:
            raise Warning("Training failed for some reason")
        print('Miniblock',new_miniblock_test.name,'from miner', miner_id, 'ready')
        miniblock_list_test.append(new_miniblock_test)
    trainging_end = time.time()
    # 出块
    new_block_test, mining_success = consensus.ensemble(miniblock_list_test, 0, False)
    if mining_success:
        print('Ensemble Block',new_block_test.name,'successfully generated')
    else:
        raise Warning('Failed to mine a new block')

    # 产生Key Block
    new_key_block_test, mining_success = consensus.mining_consensus([new_block_test],
                                                                    0,12,False,1000)
    if mining_success:
        print('Key Block',new_key_block_test.name,'successfully generated')
    else:
        raise Warning('Failed to mine a new block')

    # 将新Key Block添加到链中
    test_chain_1.add_block_direct(new_key_block_test)
    # 尝试验证该区块、添加到第二条测试链并输出结果
    validate_start = time.time()
    if consensus.valid_chain(new_key_block_test):
        test_chain_2.add_block_copy(new_key_block_test)
        # 找到最长链（假设采用最长链机制）
        depth_2 = test_chain_2.lastblock.BlockHeight()
        depth_new_block = new_key_block_test.BlockHeight()
        if depth_2 < depth_new_block:
            test_chain_2.lastblock = new_key_block_test
        validate_end = time.time()
        test_chain_2.ShowLChain()
        test_chain_2.ShowStructure1()
        print('Block metric', test_chain_2.lastblock.blockextra.metric)
        target_blockhashs = [k for k,v in test_chain_2.lastblock.blockextra.validate_list.items() \
                             if v == test_chain_2.lastblock.blockextra.metric]
        for ensemble_block in test_chain_2.lastblock.blockextra.ensemble_block_list:
            if ensemble_block.blockhead.blockhash in target_blockhashs:
                break
        print('Test set metric', consensus.validate_evaluate_miniblock(
            ensemble_block.blockextra.miniblock_list,
            ensemble_block.last.blockextra.task_queue[0],
            Task.DatasetType.TEST_SET))
        print('Validate set metric', consensus.validate_evaluate_miniblock(
            ensemble_block.blockextra.miniblock_list,
            ensemble_block.last.blockextra.task_queue[0],
            Task.DatasetType.VALIDATION_SET))

    else:
        print("Validation of a new block failed")

    print("Total Time:", time.time() - trainging_start)
    print("Training Time:", trainging_end - trainging_start)
    if validate_end:
        print("Validate Time:", validate_end - validate_start)