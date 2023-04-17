import time

import global_var

from functions import hashsha256
from chain import BlockHead, Block, Chain
from .consensus_abc import Consensus


class PoW(Consensus):

    def __init__(self):
        #self.target=global_var.get_PoW_target()
        # 严格来说target不应该出现在这里，因为这是跟共识有关的参数
        self.target = '0'
        self.ctr=0 #计数器

    def setparam(self,target):
        '''
        设置pow参数,主要是target
        '''
        self.target = target
        
    def mining_consensus(self,Blockchain:Chain,Miner_ID,isadversary,x,q):
        '''计算PoW\n
        param:
            Blockchain 该矿工维护的区块链 type:Chain
            Miner_ID 该矿工的ID type:int
            x 写入区块的内容 type:any
            qmax 最大hash计算次数 type:int
        return:
            newblock 挖出的新块 type:None(未挖出)/Block
            pow_success POW成功标识 type:Bool
        '''
        pow_success = False
        #print("mine",Blockchain)
        if Blockchain.is_empty():#如果区块链为空
            prehash = 0
            height = 0
        else:
            b_last = Blockchain.last_block()#链中最后一个块
            height = b_last.blockhead.height
            prehash = b_last.calculate_blockhash()
        currenthashtmp = hashsha256([prehash,x])    #要生成的块的哈希
        i = 0
        while i < q:
            self.ctr = self.ctr+1
            # if self._ctr>=10000000:#计数器最大值
            #     self._ctr=0
            currenthash=hashsha256([Miner_ID,self.ctr,currenthashtmp])#计算哈希
            if int(currenthash,16)<int(self.target,16):
                pow_success = True              
                blocknew=Block(''.join(['B',str(global_var.get_block_number())]),
                               BlockHead(prehash,currenthash,time.time_ns(),self.target,self.ctr,height+1,Miner_ID),
                               x,isadversary,False,global_var.get_blocksize())
                self.ctr = 0
                return (blocknew, pow_success)
            else:
                i = i+1
        return (None, pow_success)

    def valid_chain(self, lastblock: Block):
        '''验证区块链是否PoW合法\n
        param:
            lastblock 要验证的区块链的最后一个区块 type:Block
        return:
            chain_vali 合法标识 type:bool
        '''
        # xc = external.R(blockchain)
        # chain_vali = external.V(xc)
        chain_vali = True
        if chain_vali and lastblock:
            blocktmp = lastblock
            ss = blocktmp.calculate_blockhash()
            while chain_vali and blocktmp is not None:
                block_vali = self.valid_block(blocktmp)
                hash=blocktmp.calculate_blockhash()
                if block_vali and int(hash, 16) == int(ss, 16):
                    ss = blocktmp.blockhead.prehash
                    blocktmp = blocktmp.last
                else:
                    chain_vali = False
        return chain_vali

    def valid_block(self,block:Block):
        '''
        验证单个区块是否PoW合法\n
        param:
            block 要验证的区块 type:Block
        return:
            block_vali 合法标识 type:bool
        '''
        block_vali = False
        btemp = block
        target = btemp.blockhead.target
        hash = btemp.calculate_blockhash()
        if int(hash, 16) >= int(target, 16):
            return block_vali
        else:
            block_vali = True
            return block_vali