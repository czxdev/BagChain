import time
import math

import pandas as pd

import global_var
from chain import BlockHead, Block, Chain
from functions import hashsha256, hashH
from .consensus_abc import Consensus


''' R3V共识 '''
class R3V(Consensus):

    def __init__(self):
        self.target = '0'
        #self.group = 99263413 # 最接近1亿的素数 
        self.group = int('FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF43',16)
        # 115792089237316195423570985008687907853269984665640564039457584007913129639747
        # 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF43'
        # self.group = 33029
        self.primelist = pd.read_csv("prime_smaller10000000.csv").iloc

    def fastpower(self,a,n,b):
        # 快速幂取模算a^n%b
        ans = 1
        while n:
            if n&1:
                ans = ans*a%b
            a = a*a%b
            n >>= 1
        return ans%b

    def primeP(self,x:str,y:str,T):
        # 根据yx的hash计算小于2^T的质数作为证明参数
        prime_range = min(2**T,9288053) # 给定质数的上限
        prime_num = min(620421,round(prime_range/math.log(prime_range)))
        position = int(hashH([x,y]),16)%prime_num
        prime = self.primelist[position][1]
        if prime > 2**T:
            prime = 2
        return prime

    def setparam(self,target):
        #self.target = target
        self.target = hex(int(target,16)%self.group)[2:]

    def mining_consensus(self,Blockchain:Chain,Miner_ID,isadversary,x,q):
        '''计算VDF\n
        param:
            Blockchain 该矿工维护的区块链 type:Chain
            Miner_ID 该矿工的ID type:int
            x 写入区块的内容 type:any
            qmax 最大hash计算次数 type:int
        return:
            newblock 挖出的新块 type:None(未挖出)/Block
            VDF_success VDF成功标识 type:Bool
        '''
        VDF_success = False
        blocknew = None
        # s = 0
        # Blockchain.lastblock.blockhead.blockheadextra.setdefault("s",s)
        if Blockchain.lastblock.name == 'B0':
            s = 0            
            Blockchain.lastblock.blockhead.blockheadextra.setdefault("s",s)
            lastblock = Blockchain.last_block()
            height = lastblock.blockhead.height
            prehash = lastblock.calculate_blockhash()
        else:
            lastblock = Blockchain.last_block()
            s = lastblock.blockhead.blockheadextra["s"]
            height = lastblock.blockhead.height
            prehash = lastblock.calculate_blockhash()
        currenthashtmp = hashsha256([prehash,x])
        inputx = hex(int(hashH([Miner_ID,s]),16)%self.group)[2:]
        s = inputx
        # 这个s应该是作为全局共享的变量，每一轮会产生一个新的解s
        # 现在的处理方法是把s存在block里，这也符合逻辑
        # 在R3V原文中新输入s的生成使用了VRF，这里用minerID和s共同计算hash替代实现
        queryT = 1
        alpha = 1
        # 这里模拟动态难度，x是轮数，比较x和lastblock的高度
        # print("x",x)
        # print("height",lastblock.BlockHeight())
        '''
        alpha = alpha + (x - 1 - lastblock.BlockHeight())*0.2
        if int(lastblock.name[1:]) > 0.15*x:
            alpha = 1
            alpha = alpha - abs((int(lastblock.name[1:])-0.3*x))*0.5
        '''    
        while queryT <= q:
            m = int(s,16)**2%self.group
            # print("s:{}".format(m))
            # print("Target:{}".format(int(self.target,16)))
            s = hex(m)[2:]
            # print("  Query:{}".format(queryT),end="")
            
            F_s = int(hashH(s),16)%self.group
            # F_s = int(s,16)%self.group
            # print("  MinerID:{},F(s):{},Target:{}".format(Miner_ID,hex(F_s)[2:].zfill(64),hex(int(self.target,16))[2:].zfill(64)))
            
            if F_s <= round(int(self.target,16)*alpha):
                currenthash=hashsha256([Miner_ID,queryT,currenthashtmp])
                print("   Miner:{}".format(Miner_ID))
                print("   F_s:{}".format(F_s))
                print("   T:{}".format(queryT))
                print("   Target:{}".format(round(int(self.target,16)*alpha)))
                print("   Alpha:{}".format(alpha))
                VDF_success = True
                # 生成证明
                
                p = self.primeP(inputx,s,queryT)
                print("   p:{}".format(p))
                q = int(2**queryT//p)
                print("   q:{}".format(q))
                print("   x:{}".format(int(inputx,16)))
                proofPI = self.fastpower(int(inputx,16),q,self.group)
                print("   PI:{}".format(proofPI))

                blocknew=Block(''.join(['B',str(global_var.get_block_number())]),
                               BlockHead(prehash,currenthash,time.time_ns(),hex(round(int(self.target,16)*alpha)),queryT,height+1,Miner_ID),
                               x,isadversary)
                # R3V不使用块hash，故这里直接附None

                blocknew.blockhead.blockheadextra.setdefault("s",s)
                blocknew.blockhead.blockheadextra.setdefault("proofPI",proofPI)
                blocknew.blockhead.blockheadextra.setdefault("inputx",inputx)
                # 在块头的extra添加R3V需要用到的输入inputx、输出s和证明PI
                break

            queryT += 1
        return (blocknew,VDF_success)

    
    def valid_chain(self, lastblock:Block):
        '''检验链是否合法
        应return:
            合法标识    type:bool
        '''
        # R3V验证合法性其实和POW类似，验证当前块计算VDF的输入是否为上一个块的s与生成快的minerID共同hash
        chain_vali = True
        if lastblock.BlockHeight() != 0:
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
        验证单个区块是否R3V合法\n
        param:
            block 要验证的区块 type:Block
        return:
            block_vali 合法标识 type:bool
        '''
        
        # R3V 要验证两件事情：1. VDF计算结果是否正确，2. F(s)是否小于难度值
        block_vali = False
        if block.name == 'B0':
            return True
        x = block.blockhead.blockheadextra["inputx"]
        
        y = block.blockhead.blockheadextra["s"]
        pi = block.blockhead.blockheadextra["proofPI"]
        t = block.blockhead.nonce
        p = self.primeP(x,y,t)
        r = self.fastpower(2,t,p)
        x = int(x,16)
        y_0 = (self.fastpower(pi,p,self.group)*self.fastpower(x,r,self.group))%self.group
        F_s = int(hashH(y),16)%self.group

        if y_0 == int(y,16) and F_s <= int(block.blockhead.target,16):
            block_vali = True
        else:
            print(block.name)
            print("x",x)
            print("pi",pi)
            print("t",t)
            print("p",p)
            print("r",r)
            print("y_0",y_0)
            print("y",int(y,16))
            print(int(block.blockhead.target,16))

        return block_vali