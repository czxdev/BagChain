'''实现Miner类'''
import logging
from enum import Enum

import global_var
from chain import Block, Chain, BlockHead
from functions import for_name
from external import I
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

    def __init__(self, Miner_ID, qmax, target):
        '''初始化'''
        self.Miner_ID = Miner_ID #矿工ID
        self.isAdversary = False
        self.qmax = qmax
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
        self.test_set_publication_channel = [] # 测试集发布信道
        self.miniblock_storage = [] # miniblock暂存区

    def test_set_published(self, prehash, task_id):
        '''检查测试集是否已经发布，如果已经发布则返回对应的发布消息
        param:
            prehash miniblock的prehash
            task_id miniblock的task_id
        return:
            message 测试集发布消息，如果没有则返回None
        '''
        for message in self.test_set_publication_channel:
            if message[0] == prehash and message[1] == task_id:
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
        if rcvblock.blockextra.is_miniblock:
            if rcvblock not in self.miniblock_storage:
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
        # TODO 如果task_list中不止一个任务，可以改为获取费用最高的任务
        if len(self.miniblock_storage) < \
            self.Blockchain.lastblock.blockextra.task_list[0].miniblock_num:
            if self.state == self.MinerState.WAITING_MINIBLOCK:
                return (None, False)
            # TODO 如果task_list中不止一个任务，task_list[0]需要修改
            if self.test_set_published(self.Blockchain.lastblock.blockhead.blockhash,
                                       id(self.Blockchain.lastblock.blockextra.task_list[0])):
                # 已经到第二阶段，不再挖miniblock
                self.state = self.MinerState.WAITING_MINIBLOCK
                return (None, False)
            outcome, mine_success = self.consensus.train(
                self.Blockchain.lastblock, self.Miner_ID, self.isAdversary)
            if mine_success:
                self.state = self.MinerState.WAITING_MINIBLOCK
        else:
            message = self.test_set_published(
                self.miniblock_storage[0].blockhead.prehash,
                self.miniblock_storage[0].blockextra.task_id)
            # 检查测试集是否已经发布
            if message:
                outcome, mine_success = self.consensus.mining_consensus(
                    self.miniblock_storage, self.Miner_ID, self.input, self.isAdversary)
                if mine_success:
                    self.Blockchain.AddBlock(outcome)
                    self.miniblock_storage = []
                    self.state = self.MinerState.MINING_MINIBLOCK
                    self.test_set_publication_channel.remove(message)
                else:
                    self.state = self.MinerState.WAITING_MINIBLOCK
            else:
                outcome = None
                mine_success = False

        return (outcome, mine_success) # 返回结果
    
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

    def maxvalid(self):
        # 处理receive_tape中的每一项
        # output: (self.Blockchain, new_update)
        #   self.Blockchain 本地区块链
        #   new_update 本地区块链更新标识，有更新时为True
        new_update = False  # 有没有更新本地区块链
        if not self.receive_tape:
            return self.Blockchain, new_update
        
        for incoming_data in self.receive_tape:
            if incoming_data.blockextra.is_miniblock:
                if not self.consensus.valid_miniblock(incoming_data, 
                    self.test_set_published(incoming_data.blockhead.prehash, 
                                            incoming_data.blockextra.task_id)):
                    continue # 无效miniblock
                if incoming_data.last in self.Blockchain:
                    otherblock = None
                else:
                    otherblock = incoming_data.last
            else:
                otherblock = incoming_data
            if otherblock:
                if self.consensus.validate(otherblock):
                    # 把合法链的公共部分加入到本地区块链中
                    blocktmp = self.Blockchain.AddChain(otherblock)
                    # 找到最长链
                    depthself = self.Blockchain.lastblock.BlockHeight()
                    depthOtherblock = otherblock.BlockHeight()
                    if depthself < depthOtherblock:
                        self.Blockchain.lastblock = blocktmp
                        new_update = True
                        self.miniblock_storage = []
                        message = self.test_set_published(self.Blockchain.lastblock.blockhead.prehash,
                                                          self.Blockchain.lastblock.blockextra.task_id)
                        if message:
                            self.test_set_publication_channel.remove(message)
                        self.state = self.MinerState.MINING_MINIBLOCK
                else:
                    logger.error("validation of block %s failure", 
                                otherblock.name)  # 验证失败
                    continue
            if incoming_data.blockextra.is_miniblock:
                if incoming_data.last.blockhead.blockhash == \
                self.Blockchain.lastblock.blockhead.blockhash:
                    self.miniblock_storage.append(incoming_data)

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

