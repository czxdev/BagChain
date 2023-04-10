'''实现Miner类'''
import logging
from enum import Enum

import global_var
from chain import Block, Chain, BlockHead
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
        PENDING_FOR_VALIDATION = 3 # 区块已经产生，等待验证当前高度区块
   
    def __init__(self, Miner_ID, q, target):
        '''初始化'''
        self.Miner_ID = Miner_ID #矿工ID
        self.isAdversary = False
        self.q = q
        self.Blockchain = Chain()   # 维护的区块链
        #共识相关
        self.consensus = for_name(global_var.get_consensus_type())()    # 共识
        self.consensus.setparam()                                 # 设置共识参数
        #输入内容相关
        self.input = 0          # 要写入新区块的值
        self.input_tape = []
        #接收链相关
        self.receive_tape = []
        self.receive_history= [] #保留最近接收到的3个块
        self.buffer_size = 3
        #网络相关
        self.neighbor_list = []
        self.otherchain = Chain()
        self.processing_delay=0    #处理时延
        #共识协议相关
        self.state = self.MinerState.MINING_MINIBLOCK
        self.dataset_publication_channel = [] # 数据集发布通道
        self.miniblock_storage = [] # miniblock暂存区
        self.miniblock_pending_list = [] # 存储接收到的miniblock（无论是否是获胜区块）

    def dataset_published(self, prehash:str, task_id, dataset_type:Task.DatasetType):
        '''检查测试集是否已经发布，如果已经发布则返回对应的发布消息
        param:
            prehash miniblock的prehash
            task_id miniblock的task_id
        return:
            message 测试集发布消息，如果没有则返回None
        '''
        for message in self.dataset_publication_channel:
            if message[0] is dataset_type and message[1] == prehash \
                and message[2] == task_id:
                return message
        return None

    def set_Adversary(self, isAdversary:bool):
        '''
        设置是否为对手节点
        isAdversary=True为对手节点
        '''
        self.isAdversary = isAdversary

    def isInLocalChain(self,block:Block):
        '''Check whether a block is in local chain,
        param: block: The block to be checked
        return: flagInLocalChain: Flag whether the block is in local chain.If not, return False; else True.'''
        if block not in self.Blockchain:
            return False
        else:
            return True

    def receiveBlock(self,rcvblock:Block):
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
        if rcvblock.blockextra.blocktype is rcvblock.BlockType.MINIBLOCK:
            if rcvblock not in self.miniblock_storage and \
                rcvblock not in self.miniblock_pending_list:
                self.receive_tape.append(rcvblock)
                return True
        else:
            if rcvblock not in self.Blockchain:
                self.receive_tape.append(rcvblock)
                return True

        return False

    def sendBlock(self, to,sendblock:Block):
        for nb in self.neighbor_list:
            pass

    def Mining(self):
        '''挖矿\n
        return:(outcome, mine_success)
            outcome 挖出的新区块或miniblock,没有就返回none type:Block/None
            mine_success 挖矿成功标识 type:Bool
        '''
        outcome = None
        mine_success = False
        prehash = self.Blockchain.lastblock.blockhead.blockhash
        taskid = id(self.Blockchain.lastblock.blockextra.task_queue[0])
        
        if self.state is self.MinerState.MINING_MINIBLOCK:
            if self.dataset_published(prehash, taskid,
                                      Task.DatasetType.VALIDATION_SET):
                self.state = self.MinerState.PENDING_FOR_VALIDATION
            elif self.dataset_published(prehash, taskid,
                                        Task.DatasetType.TEST_SET):
                self.state = self.MinerState.WAITING_MINIBLOCK
            else: # 还没有发布测试集
                if validate_metric := self.Blockchain.lastblock.blockextra.validate_metric \
                    or self.Blockchain.lastblock.isGenesis:
                    outcome, mine_success = self.consensus.train(
                        self.Blockchain.lastblock, self.Miner_ID, 
                        validate_metric, self.isAdversary)
                elif not self.Blockchain.lastblock.isGenesis:
                    raise Warning("Enter MINING_MINIBLOCK mode before validate lastblock")
                if mine_success:
                    self.miniblock_storage.append(outcome)
                    self.state = self.MinerState.WAITING_MINIBLOCK
        elif self.state is self.MinerState.WAITING_MINIBLOCK:
            if self.dataset_published(prehash, taskid,
                                      Task.DatasetType.VALIDATION_SET):
                self.state = self.MinerState.PENDING_FOR_VALIDATION
            elif self.dataset_published(prehash, taskid,
                                      Task.DatasetType.TEST_SET):
                outcome, mine_success = self.consensus.mining_consensus(
                    self.miniblock_storage, self.Miner_ID, self.input, self.isAdversary)
                if mine_success:
                    self.Blockchain.AddBlock(outcome)
                    self.state = self.MinerState.PENDING_FOR_VALIDATION
        elif self.state is self.MinerState.PENDING_FOR_VALIDATION:
            if not self.dataset_published(prehash, taskid,
                                      Task.DatasetType.TEST_SET):
                raise Warning("Enter PENDING_FOR_VALIDATION before any test set published")
        
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
            IsValid=self.consensus.validate(self.Blockchain.lastblock)
            if IsValid:
                print('Miner', self.Miner_ID, 'self_blockchain validated\n')
            else:
                print('Miner', self.Miner_ID, 'self_blockchain wrong\n')
        else:
            IsValid = self.consensus.validate(blockchain)
            if not IsValid:
                print('blockchain wrong\n')
        return IsValid

    def validate_chain_metric(self, block:Block):
        '''
        评估链上validate_metric为None的区块
        判断测试集是否发布并评估、记录验证集性能validate_metric
        param:
            block: 待检验链的尾部 Type: Block
        '''
        if block.blockextra.blocktype is Block.BlockType.MINIBLOCK:
            raise Warning("Expect a Block")
        blocktmp = block
        block_with_no_validation_set = 0
        while blocktmp and not blocktmp.isGenesis and \
            blocktmp.blockextra.validate_metric is None:
            # 从后向前遍历所有validate_metric为None的区块
            if self.dataset_published(blocktmp.blockhead.prehash,
                                      blocktmp.blockextra.task_id,
                                      Task.DatasetType.VALIDATION_SET):
                # 验证集已经发布
                task_blocktmp = blocktmp.last.blockextra.task_queue[0]
                blocktmp.blockextra.validate_metric = \
                        self.consensus.validate_evaluate_miniblock(
                        blocktmp.blockextra.miniblock_list, task_blocktmp,
                        Task.DatasetType.VALIDATION_SET)
            else:
                block_with_no_validation_set += 1
            blocktmp = blocktmp.last
        if block_with_no_validation_set > 1:
            raise Warning("Only the tail of a chain may exist without any\
                          validation set published")
        
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
                if self.consensus.validate(incoming_data):
                    # 把合法链的公共部分加入到本地区块链中
                    blocktmp = self.Blockchain.AddChain(incoming_data)
                    # 若验证集已经发布先评估指标，记录在validate_metric中
                    self.validate_chain_metric(blocktmp)
                    # 找到最长链，判断最长链的末端验证集是否已经发布
                    # 由于lastblock的验证集不一定已经发布，需要将当前块的高度+1
                    # depthself = self.Blockchain.lastblock.BlockHeight() + 1
                    depthself = self.Blockchain.lastblock.BlockHeight()
                    depthOtherblock = incoming_data.BlockHeight()
                    if depthself < depthOtherblock:
                        if blocktmp.blockextra.validate_metric is not None:
                            # 有高度更高的链，并且验证集已经发布
                            self.Blockchain.lastblock = blocktmp
                        elif depthself+1 < depthOtherblock:
                            # 高度+2的任务还没有发布验证集，lastblock为下一高度的获胜块
                            # 由于validate_chain_metric进行了检查，blocktmp.last一定已经经过验证
                            self.Blockchain.lastblock = blocktmp.last
                        else: # 下一高度的块，验证集还没有发布，没有办法评估验证集性能
                            continue
                        new_update = True
                        # 将当前miniblock_storage的内容放入pending list留存一个区块高度
                        self.miniblock_pending_list.extend(self.miniblock_storage)
                        self.miniblock_storage = []
                        self.state = self.MinerState.MINING_MINIBLOCK
                    
                    # 或者找到区块高度相同但是性能更好的链，但需要保证任务相同才能比较性能指标
                    if  depthself == depthOtherblock \
                        and incoming_data.blockextra.task_id == \
                            self.Blockchain.lastblock.blockextra.task_id \
                        and incoming_data.blockextra.validate_metric is not None \
                        and incoming_data.blockextra.validate_metric > \
                            self.Blockchain.lastblock.blockextra.validate_metric:
                        
                        self.Blockchain.lastblock = blocktmp
                        new_update = True
                        # 将当前miniblock_storage的内容放入pending list留存一个区块高度
                        self.miniblock_pending_list.extend(self.miniblock_storage)
                        self.miniblock_storage = []
                        self.state = self.MinerState.MINING_MINIBLOCK
                else:
                    logger.error("validation of block %s failure", 
                                incoming_data.name)  # 验证失败
                    continue
        
        # 查找当前高度上是否有区块的验证集已经发布
        # 正常情况下一个高度上客户仅为一个任务发布数据集
        # 当前高度上区块列表
        blocks_current_height = self.Blockchain.lastblock.last.next \
                                if not self.Blockchain.lastblock.isGenesis \
                                else [self.Blockchain.lastblock]
        for block in blocks_current_height:
            prehash = block.blockhead.blockhash
            current_task = block.blockextra.task_queue[0]
            if self.dataset_published(prehash, id(current_task),
                                      Task.DatasetType.VALIDATION_SET):
                # 验证集发布后开始判断获胜区块
                # 寻找当前高度上next中区块最多的块
                fork_list = block.next
                if len(fork_list) == 0:
                    logger.warning("validation set published before any valid block received")
                    break
                best_block = fork_list[0]
                best_block.blockextra.validate_metric = self.consensus.validate_evaluate_miniblock(
                                best_block.blockextra.miniblock_list, current_task,
                                Task.DatasetType.VALIDATION_SET)
                optimal_metric = best_block.blockextra.validate_metric
                # 找到验证集性能最好的块
                for block in fork_list[1:]:
                    metric_temp = self.consensus.validate_evaluate_miniblock(
                        block.blockextra.miniblock_list, current_task,
                        Task.DatasetType.VALIDATION_SET)
                    block.blockextra.validate_metric = metric_temp
                    if  metric_temp > optimal_metric:
                        best_block = block
                        optimal_metric = metric_temp
                self.Blockchain.lastblock = best_block
                new_update = True
                # 将当前miniblock_storage的内容放入pending list留存一个区块高度
                self.miniblock_pending_list.extend(self.miniblock_storage)
                self.miniblock_storage = []
                self.state = self.MinerState.MINING_MINIBLOCK
                break # 一个高度上客户仅为一个任务发布数据集
                      # 且任务队列已经保证当前高度上任务已经在若干个高度前确定

        new_list = []
        for miniblock in self.miniblock_pending_list:
            if miniblock.last in self.Blockchain:
                if miniblock.last.blockhead.blockhash == \
                self.Blockchain.lastblock.blockhead.blockhash and \
                    miniblock not in self.miniblock_storage:
                    # 只有获胜区块后续的miniblock可以放入miniblock_storage
                    # 这些区块不会留在pending list
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

        return self.Blockchain, new_update
    
    def BackboneProtocol(self, round) -> Block:
        '''执行核心共识协议'''
        chain_update, update_index = self.maxvalid()
        # if input contains READ:
        # write R(Cnew) to OUTPUT() 这个output不知道干什么用的
        self.input = I(round, self.input_tape)  # I function
        #print("outer2",honest_miner.input)
        newblock, mine_success = self.Mining()
        #print("outer3",honest_miner.input)
        if update_index or mine_success:  # Cnew != C
            return newblock
        else:
            return None  #  如果没有更新 返回空告诉environment回合结束


if __name__ =='__main__':
    global_var.__init__()
    miner1=Miner(1,2,3)
    miner1.receive_buffer=[0,4,6]
    list=[0,1,2,3,4,5,6]
    miner1.receiveBlock(9)
    print(miner1.receive_buffer)
    print(miner1.receive_tape)
    miner1.receiveBlock(Block())
    print(miner1.receive_buffer)
    print(miner1.receive_tape)

