'''实现Miner类'''
import logging
from enum import Enum

import global_var
from chain import Block, Chain, BlockList
from consensus import Consensus
from functions import for_name
from external import I
from task import Task
##区块链不是链表，用树结构
##Q同步，认为有一个时延
##理论仿真器
##评估从文章里面3个指标
##互相怎么传参的

logger = logging.getLogger(__name__)

class Miner(object):
    '''Miner代表了网络中的一个个能够自主维护区块链的实体，执行包括接收、转发、生成区块在内的行动'''
    class MinerState(Enum):
        '''描述矿工状态的类'''
        MINING_MINIBLOCK = 1 # 收到区块开始尝试产生miniblock
        WAITING_MINIBLOCK = 2 # 等待来自其他矿工的miniblock
        WAITING_ENSEMBLE_BLOCK = 3 # 等待验证集发布接收来自其他矿工的Ensemble Block
   
    def __init__(self, Miner_ID, q, target):
        '''初始化'''
        self.Miner_ID = Miner_ID #矿工ID
        self.isAdversary = False
        self.q = q
        self.Blockchain = Chain()   # 维护的区块链
        #共识相关
        self.consensus:Consensus = for_name(global_var.get_consensus_type())()    # 共识
        self.consensus.setparam(target)                                 # 设置共识参数
        #输入内容相关
        self.input = 0          # 要写入新区块的值
        self.input_tape = []
        #接收链相关
        self.receive_tape:BlockList = []
        self.receive_history= [] #保留最近接收到的3个块
        self.buffer_size = 3
        #网络相关
        self.neighbor_list = []
        self.processing_delay=0    #处理时延
        #共识协议相关
        self.state = self.MinerState.MINING_MINIBLOCK
        self.dataset_publication_channel = [] # 数据集发布通道
        self.miniblock_storage:BlockList = [] # miniblock暂存区
        self.miniblock_pending_list:BlockList = [] # 存储接收到的miniblock（非获胜Key Block）
        self.ensemble_block_storage:BlockList = [] # Ensemble Block暂存区
        self.ensemble_block_pending_list:BlockList = [] # 存储接收到的Ensemble Blokc（非获胜Key Block）


    def dataset_published(self, prehash:str, task_id, dataset_type:Task.DatasetType, height):
        '''检查测试集是否已经发布，如果已经发布则返回对应的发布消息
        param:
            prehash 上一高度获胜区块的哈希
            task_id 当前高度执行的任务
            height 上一获胜区块所在区块高度
        return:
            message 测试集发布消息，如果没有则返回None
        '''
        for message in self.dataset_publication_channel:
            # if message[0] is dataset_type \
            if message[0] is dataset_type and message[3] == height \
                            and message[2] == task_id:
                return message
        return None

    def set_adversary(self, isAdversary:bool):
        '''
        设置是否为对手节点
        isAdversary=True为对手节点
        '''
        self.isAdversary = isAdversary

    def is_in_local_chain(self,block:Block):
        '''Check whether a block is in local chain,
        param: block: The block to be checked
        return: flagInLocalChain: Flag whether the block is in local chain.If not, return False; else True.'''
        if block.blockextra.blocktype is block.BlockType.MINIBLOCK:
            if block in self.miniblock_storage or \
                block in self.miniblock_pending_list:
                return True
        elif block.blockextra.blocktype is block.BlockType.KEY_BLOCK:
            if block in self.Blockchain:
                return True
        elif block.blockextra.blocktype is block.BlockType.BLOCK:
            if block in self.ensemble_block_storage or \
                block in self.ensemble_block_pending_list:
                return True
        return False


    def receive_block(self,rcvblock:Block):
        '''Interface between network and miner.
        Append the rcvblock(have not received before) to receive_tape, 
        and add to local chain in the next round. 
        param: 
            rcvblock: The block received from network. Type: Block
        return: 
            flagNotRecBlockRecently: Flag representing whether the rcvblock is 
            already in local chain or in miniblock_storage. 
            If not, return True; else False.
        '''
        if not self.is_in_local_chain(rcvblock):
            self.receive_tape.append(rcvblock)
            return True
        return False


    def mining(self):
        '''挖矿\n
        return:(outcome, mine_success)
            outcome 挖出的新区块或miniblock,没有就返回none type:Block/None
            mine_success 挖矿成功标识 type:Bool
        '''
        outcome = None
        mine_success = False
        prehash = self.Blockchain.lastblock.blockhead.blockhash
        taskid = id(self.Blockchain.lastblock.blockextra.task_queue[0])
        height = self.Blockchain.lastblock.blockhead.height
        
        if self.state is self.MinerState.MINING_MINIBLOCK:
            if self.dataset_published(prehash, taskid,
                                      Task.DatasetType.VALIDATION_SET,
                                      height):
                self.state = self.MinerState.WAITING_ENSEMBLE_BLOCK
            elif self.dataset_published(prehash, taskid,
                                        Task.DatasetType.TEST_SET,
                                        height):
                self.state = self.MinerState.WAITING_MINIBLOCK
            else: # 还没有发布测试集
                outcome, mine_success = self.consensus.train(
                    self.Blockchain.lastblock, self.Miner_ID, self.isAdversary)
                if mine_success:
                    self.miniblock_storage.append(outcome)
                    self.state = self.MinerState.WAITING_MINIBLOCK
        elif self.state is self.MinerState.WAITING_MINIBLOCK:
            if self.dataset_published(prehash, taskid,
                                      Task.DatasetType.VALIDATION_SET,
                                      height):
                self.state = self.MinerState.WAITING_ENSEMBLE_BLOCK
            elif self.dataset_published(prehash, taskid,
                                      Task.DatasetType.TEST_SET,
                                      height):
                if len(self.miniblock_storage) > 0:
                    outcome, mine_success = self.consensus.ensemble(
                        self.miniblock_storage, self.Miner_ID, self.isAdversary)
                if mine_success:
                    self.ensemble_block_storage.append(outcome)
                    self.state = self.MinerState.WAITING_ENSEMBLE_BLOCK
        elif self.state is self.MinerState.WAITING_ENSEMBLE_BLOCK:
            if not self.dataset_published(prehash, taskid,
                                      Task.DatasetType.TEST_SET,
                                      height):
                raise Warning("Enter WAITING_ENSEMBLE_BLOCK before any test set published")
            if self.dataset_published(prehash, taskid,
                                      Task.DatasetType.VALIDATION_SET,
                                      height):
                if len(self.ensemble_block_storage) > 0:
                    outcome, mine_success = self.consensus.mining_consensus(
                        self.ensemble_block_storage, self.Miner_ID, self.input,
                        self.isAdversary, self.q)
                if mine_success:
                    self.Blockchain.add_block_direct(outcome)
                    self.miniblock_pending_list.extend(self.miniblock_storage)
                    self.miniblock_storage = []
                    self.ensemble_block_pending_list.extend(self.ensemble_block_storage)
                    self.ensemble_block_storage = []
                    self.state = self.MinerState.MINING_MINIBLOCK
        return (outcome, mine_success)
    
    def ValiChain(self, blockchain: Chain = None):
        '''
        检查是否满足共识机制\n
        相当于原文的validate
        输入:
            blockchain 要检验的区块链 type:Chain
            若无输入,则检验矿工自己的区块链
        输出:
            IsValid 检验成功标识 type:bool
        '''
        if blockchain==None:#如果没有指定链则检查自己
            IsValid=self.consensus.valid_chain(self.Blockchain.lastblock)
            if IsValid:
                print('Miner', self.Miner_ID, 'self_blockchain validated\n')
            else:
                print('Miner', self.Miner_ID, 'self_blockchain wrong\n')
        else:
            IsValid = self.consensus.valid_chain(blockchain)
            if not IsValid:
                print('blockchain wrong\n')
        return IsValid
        
    def maxvalid(self):
        # 处理receive_tape中的每一项
        # output: (self.Blockchain, new_update)
        #   self.Blockchain 本地区块链
        #   new_update 本地区块链更新标识，有更新时为True
        new_update = False  # 有没有更新本地区块链
        
        for incoming_data in self.receive_tape:
            if incoming_data.blockextra.blocktype is incoming_data.BlockType.MINIBLOCK:
                if not self.consensus.valid_miniblock(incoming_data):
                    continue # 无效miniblock
                if incoming_data.last in self.Blockchain:
                    if incoming_data.last.blockhead.blockhash == \
                    self.Blockchain.lastblock.blockhead.blockhash:
                        self.miniblock_storage.append(incoming_data) 
                        continue
                # 缓存miniblock
                if incoming_data.blockhead.height > \
                    self.Blockchain.lastblock.blockhead.height:
                    # 利用块头中的高度信息，保留比本地区块链更高的miniblock
                    self.miniblock_pending_list.append(incoming_data)
                    logger.info("%s enter pending list in Miner %d",
                                    incoming_data.name, self.Miner_ID)
            elif incoming_data.blockextra.blocktype is incoming_data.BlockType.BLOCK:
                if not self.consensus.valid_block(incoming_data):
                    continue # 无效Ensenble Block
                if incoming_data.last in self.Blockchain:
                    if incoming_data.last.blockhead.blockhash == \
                    self.Blockchain.lastblock.blockhead.blockhash:
                        self.ensemble_block_storage.append(incoming_data) 
                        continue
                # 缓存Ensemble Block
                if incoming_data.blockhead.height > \
                    self.Blockchain.lastblock.blockhead.height:
                    # 利用块头中的高度信息，保留比本地区块链更高的Ensemble Block
                    self.ensemble_block_pending_list.append(incoming_data)
                    logger.info("%s enter pending list in Miner %d",
                                    incoming_data.name, self.Miner_ID)
            elif incoming_data.blockextra.blocktype is incoming_data.BlockType.KEY_BLOCK:
                copylist, insert_point = self.consensus.valid_partial(incoming_data, self.Blockchain)
                if copylist is not None:
                    # 把合法链的公共部分加入到本地区块链中
                    blocktmp = self.Blockchain.insert_block_copy(copylist, insert_point)
                    # 找到最长链，判断最长链的末端验证集是否已经发布
                    # 或者找到区块高度相同但是性能更好的链，但需要保证任务相同才能比较性能指标
                    depthself = self.Blockchain.lastblock.BlockHeight()
                    depthOtherblock = incoming_data.BlockHeight()
                    taskid = incoming_data.blockextra.task_id
                    prehash = incoming_data.blockhead.blockhash
                    if depthself < depthOtherblock or depthself == depthOtherblock \
                        and incoming_data.blockextra.task_id == \
                            self.Blockchain.lastblock.blockextra.task_id \
                        and incoming_data.blockextra.metric > \
                            self.Blockchain.lastblock.blockextra.metric \
                        and not self.dataset_published(prehash, taskid, Task.DatasetType.TEST_SET, depthOtherblock) \
                        and not self.dataset_published(prehash, taskid, Task.DatasetType.VALIDATION_SET, depthOtherblock):
                        self.Blockchain.lastblock = blocktmp
                        new_update = True
                        # 将当前存储的miniblock/ensemble block放入pending list留存一个区块高度
                        self.miniblock_pending_list.extend(self.miniblock_storage)
                        self.ensemble_block_pending_list.extend(self.ensemble_block_storage)
                        self.miniblock_storage = []
                        self.ensemble_block_storage = []
                        self.state = self.MinerState.MINING_MINIBLOCK
                else:
                    logger.error("validation of block %s failure", 
                                incoming_data.name)  # 验证失败
                    continue

        new_list = []
        for miniblock in self.miniblock_pending_list:
            if miniblock.last in self.Blockchain:
                if miniblock.last.blockhead.blockhash == \
                self.Blockchain.lastblock.blockhead.blockhash and \
                    miniblock not in self.miniblock_storage:
                    # 只有获胜Key Block后续的miniblock可以放入miniblock_storage
                    # 这些miniblock不会留在pending list
                    self.miniblock_storage.append(miniblock)
                    logger.info("%s moved from pending list to storage in Miner %d",
                                 miniblock.name, self.Miner_ID)
                    continue
            if miniblock.blockhead.height >= self.Blockchain.lastblock.blockhead.height:
                # 丢弃低于本地链高度的miniblock
                # 保留miniblock接收记录，避免receiveBlock重复接收同一miniblock
                new_list.append(miniblock)
        dropped_miniblock_num = len(self.miniblock_pending_list)-len(new_list)
        if dropped_miniblock_num > 0:
            logger.info("%d miniblock(s) dropped from pending list of Miner %d",
                        dropped_miniblock_num, self.Miner_ID)
        self.miniblock_pending_list = new_list

        new_list = []
        for ensemble_block in self.ensemble_block_pending_list:
            if ensemble_block.last in self.Blockchain:
                if ensemble_block.last.blockhead.blockhash == \
                self.Blockchain.lastblock.blockhead.blockhash and \
                    ensemble_block not in self.ensemble_block_storage:
                    # 只有获胜Key Block后续的Ensemble Block可以放入ensemble_block_storage
                    # 这些ensemble_block不会留在pending list
                    self.ensemble_block_storage.append(ensemble_block)
                    logger.info("%s moved from pending list to storage in Miner %d",
                                 ensemble_block.name, self.Miner_ID)
                    continue
            if ensemble_block.blockhead.height >= self.Blockchain.lastblock.blockhead.height:
                # 丢弃低于本地链高度的Ensemble Block
                # 保留Ensemble Block接收记录，避免receiveBlock重复接收同一Ensemble Block
                new_list.append(ensemble_block)
        dropped_ensemble_block_num = len(self.ensemble_block_pending_list)-len(new_list)
        if dropped_ensemble_block_num > 0:
            logger.info("%d ensemble block(s) dropped from pending list of Miner %d",
                        dropped_ensemble_block_num, self.Miner_ID)
        self.ensemble_block_pending_list = new_list

        return self.Blockchain, new_update
    
    def BackboneProtocol(self, round) -> Block:
        '''执行核心共识协议'''
        chain_update, update_index = self.maxvalid()
        # if input contains READ:
        # write R(Cnew) to OUTPUT() 这个output不知干什么用的道
        self.input = I(round, self.input_tape)  # I function
        #print("outer2",honest_miner.input)
        newblock, mine_success = self.mining()
        #print("outer3",honest_miner.input)
        if update_index or mine_success:  # Cnew != C
            return newblock
        else:
            return None  #  如果没有更新 返回空告诉environment回合结束
        
    # def ValiChain(self, blockchain: Chain = None):
    #     '''
    #     检查是否满足共识机制\n
    #     相当于原文的validate
    #     输入:
    #         blockchain 要检验的区块链 type:Chain
    #         若无输入,则检验矿工自己的区块链
    #     输出:
    #         IsValid 检验成功标识 type:bool
    #     '''
    #     if blockchain is None:#如果没有指定链则检查自己
    #         IsValid=self.consensus.valid_chain(self.Blockchain.lastblock)
    #         if IsValid:
    #             print('Miner', self.Miner_ID, 'self_blockchain validated\n')
    #         else:
    #             print('Miner', self.Miner_ID, 'self_blockchain wrong\n')
    #     else:
    #         IsValid = self.consensus.valid_chain(blockchain)
    #         if not IsValid:
    #             print('blockchain wrong\n')
    #     return IsValid