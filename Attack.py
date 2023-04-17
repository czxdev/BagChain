import copy
import random
import time
from typing import List
from abc import ABCMeta, abstractmethod

import global_var
from chain import Block, Chain
from external import I
from functions import for_name
from miner import Miner


def get_time(f):

    def inner(*arg,**kwarg):
        s_time = time.time()
        res = f(*arg,**kwarg)
        e_time = time.time()
        print('耗时：{}秒'.format(e_time - s_time))
        return res
    return inner


class Attack(metaclass=ABCMeta): 

    @abstractmethod
    def excute(self):
        '''执行attack'''
        pass

 
class Selfmining(Attack):

    def __init__(self, globalchain:Chain, target, network, Adversary:List[Miner], num_rounds) -> None:
        self.Adver = Adversary
        self.num_rounds = num_rounds

        self.mine_chain = copy.deepcopy(globalchain)

        self.honest_chain = copy.deepcopy(globalchain)

        self.base_chain = copy.deepcopy(globalchain)


        # 不再补充设置q和target
        self.q = self.Adver[0].q
        self.consensus = for_name(global_var.get_consensus_type())()
        self.consensus.setparam(target)
        # 从环境中提取共识 这里选择在Adver集团建立一个统一的共识对象 共享挖掘进度
        self.round = 1
        self.network = network
        self.globalchain = globalchain
        self.tolerant = 0 # 链同长的忍耐指数
        self.broad_block = None
        self.excute_type = 'Normal' # 'Probability'  'Normal'
        self.tolerant_len = 2
        self.lastexcute = ''
        self.attacklog = []
        self.honest2base_refresh = True # base链是否更新成honest链，默认初始是一致的
        
        #
   
    def Adopt(self):
        receive_list_cur_round = []
        for attacker in self.Adver:
            # 将Adver从诚实节点中接受到的chain进行收集
            ''' 模仿诚实节点对adver集团内的attacker进行maxvalid '''
            for block in attacker.receive_tape:
                if attacker.consensus.valid_chain(block):
                    # attacker也要对接收到的block进行验证
                    # 如果验证成功添加到attacker的本地链上
                    blocktmp = attacker.Blockchain.add_block_copy(block)
                    depthself = attacker.Blockchain.lastblock.BlockHeight()
                    depthOtherblock = block.BlockHeight()
                    if depthself < depthOtherblock:
                        attacker.Blockchain.lastblock = blocktmp
            receive_list_cur_round.append(attacker.Blockchain)
            attacker.receive_tape = []  
            # 将每个attacker的chain加到receive_list_cur_round，最后再择优

        chain_new_honest = False
        height = -1
        while receive_list_cur_round:
            chain_cur = receive_list_cur_round.pop(0)
            if height < chain_cur.lastblock.BlockHeight():
                height = chain_cur.lastblock.BlockHeight()
                best_chain = chain_cur
        # 以Adver集团收到的所有链为基础择出最优链，判断准则是最长的链
        #print(best_chain.lastblock.BlockHeight(),self.honest_chain.lastblock.BlockHeight())
        #print(id(best_chain),id(self.honest_chain))
        if best_chain.lastblock.BlockHeight() > self.honest_chain.lastblock.BlockHeight():
            self.honest_chain = copy.deepcopy(best_chain)
            # self.honest_chain.AddChain(best_chain.lastblock)
            #print(id(best_chain),id(self.honest_chain))
            chain_new_honest = True
            # 这里就不考虑深拷贝的事情了，毕竟Adver拥护的链是都一条，本质上是Attacker身上的链，已经深拷贝处理过了

            # for attacker in self.Adver:
            #     attacker.Blockchain.AddChain(self.honest_chain.lastblock)
                # 再将集团旗下的attacker的链都进行更新一下，保持一致
        '''
        if self.honest_chain.lastblock.BlockHeight() > self.mine_chain.lastblock.BlockHeight():
            chain_new_mine = True
            self.base_chain = self.honest_chain
            print("攻击目标更新")
        else:
            chain_new_mine = False
        '''    
        if chain_new_honest:
            self.honest2base_refresh = False
            # 如果从attacker中有更新链，base链还没有更新，置为False
        else:
            self.honest2base_refresh = True
        return chain_new_honest #, chain_new_mine 
        # 这里主要看Adver集团的chain有没有更新，返回一个chain_new标志
        # chain_new_mine 主要是看

    @get_time
    def Adopt_2(self):
        receive_list_cur_round = []
        height = -1
        chain_new_honest = False
        for attacker in self.Adver:
            attacker.maxvalid()
            I(self.round,attacker.input_tape)
            if attacker.Blockchain.lastblock.BlockHeight() > height:
                bestchain = attacker.Blockchain
            attacker.input_tape = []
            attacker.receive_tape = []
        if self.honest_chain.lastblock.BlockHeight() < bestchain.lastblock.BlockHeight():
            self.honest_chain = bestchain
            chain_new_honest = True
            self.honest2base_refresh = False
        return chain_new_honest

    def Advermine_input(self):
        ''' Adver集团产生挖矿时的input '''
        for attacker in self.Adver:
            attacker.input = I(self.round, attacker.input_tape)
            attacker.input_tape = []
        input = self.Adver[0].input
        return input


    def Advermine(self):
        ''' Adver集团挖 '''
        mine_power = len(self.Adver) * self.q
        Miner_ID = self.Adver[0].Miner_ID
        input = self.Advermine_input()
        newblock, mine_success = self.consensus.mining_consensus(self.base_chain,Miner_ID,\
            True, input, mine_power)
        #if self.consensus.ctr == 0:
            # print("self block")
            
        if mine_success is True:
            # self.mine_chain.AddBlock(newblock)
            # self.mine_chain.lastblock = newblock
            self.attacklog.append([self.round,''.join(['Selfly mine ',newblock.name,'\n',\
            'honest chain:',str(self.honest_chain.lastblock.BlockHeight()),'\n',\
            'self chain:',str(self.mine_chain.lastblock.BlockHeight()),'\n',\
            'base chain:',str(self.base_chain.lastblock.BlockHeight())])])
            self.base_chain.add_block_direct(newblock)
            #self.base_chain.lastblock = newblock
            # 更新挖掘的base_chain
            self.mine_chain = copy.deepcopy(self.base_chain)
            #self.mine_chain.AddChain(newblock)
            # 挖掘的结果更新到mine_chain
            self.globalchain.add_block_copy(newblock)
        return newblock, mine_success  # 返回挖出的区块，


    def broadcast_network(self):
        if self.mine_chain.lastblock != self.broad_block:
            self.network.access_network(self.mine_chain.lastblock, self.Adver[0].Miner_ID,self.round)
            self.broad_block = self.mine_chain.lastblock
            broadcast = False
        else:
            broadcast = True
        return broadcast


    def Wait(self):
        # print("Wait")
        pass

    
    def Giveup(self):
        #print("Adversary Giveup", end='', flush=True)
        self.attacklog.append([self.round,''.join(['Give up\n',\
        'honest chain:',str(self.honest_chain.lastblock.BlockHeight()),'\n',\
        'self chain:',str(self.mine_chain.lastblock.BlockHeight()),'\n',\
        'base chain:',str(self.base_chain.lastblock.BlockHeight())])])
        self.base_chain = copy.deepcopy(self.honest_chain)
        self.mine_chain = copy.deepcopy(self.honest_chain)
        #self.base_chain.AddChain(self.honest_chain.lastblock)
        #self.mine_chain.AddChain(self.honest_chain.lastblock) 
        self.printchainlen()
        self.honest2base_refresh = True


    def Match(self):
        ''' 即使链同长 Adver集团还是选择将链公布 '''
        is_broadcast = self.broadcast_network()
        if is_broadcast:
            self.Wait()
        else:
            #print("Adversary match", end='', flush=True)
            self.printchainlen()
            self.attacklog.append([self.round,''.join(['Match\n',\
            'honest chain:',str(self.honest_chain.lastblock.BlockHeight()),'\n',\
            'self chain:',str(self.mine_chain.lastblock.BlockHeight()),'\n',\
            'base chain:',str(self.base_chain.lastblock.BlockHeight())])])


    def Override(self):
        if self.mine_chain.lastblock.BlockHeight() > self.honest_chain.lastblock.BlockHeight():
            is_broadcast = self.broadcast_network()
            if is_broadcast:
                self.Wait()
                override = False
            else:
                #print("Adversary override", end='', flush=True)
                self.attacklog.append([self.round,''.join(['Override\n',\
                'honest chain:',str(self.honest_chain.lastblock.BlockHeight()),'\n',\
                'self chain:',str(self.mine_chain.lastblock.BlockHeight()),'\n',\
                'base chain:',str(self.base_chain.lastblock.BlockHeight())])])     
                self.printchainlen()        
                override = True
        else:
            override = False
        return override
    
    def printchainlen(self):
        #print("self chain:",self.mine_chain.lastblock.BlockHeight())
        #print("honest chain:",self.honest_chain.lastblock.BlockHeight())
        #print("base chain:",self.base_chain.lastblock.BlockHeight())
        return 0

    def attacklog2txt(self):
        RESULT_PATH = global_var.get_result_path()
        with open(RESULT_PATH / 'Attack_log.txt','a') as f:
            print('Attack Type: ',self.excute_type,'\n',file=f)
            while self.attacklog:
                log = self.attacklog.pop(0)
                print('Round:',log[0],file=f)
                print(log[1],'\n',file=f)


    def Probability_excute(self):
        honest_update = self.Adopt()
        '''
        honest_update 反映Adver集团认定的来自诚实节点的链 honest_chain 有无更新
        # mine_update 反映Adver集团挖掘攻击基于的链 base_chain 有无更新 #
        Adopt 干了以下三件事：
        1. 每个attacker执行类似maxvalid更新自己链
        2. 1更新后attacker身上的链是来自诚实网络的 所有attacker分享这些链 比较择优出最佳的诚实链 
            更新到集团认可的honest_chain
        3. 将2的链更新到所有attacker身上

        '''
        if honest_update:
            print("更新诚实链")
        if self.mine_chain.lastblock.BlockHeight() > self.honest_chain.lastblock.BlockHeight():
            newblock, mine_success = self.Advermine()
            if random.uniform(0 + self.tolerant, 1) <= 0.8:
                self.Override()
                self.tolerant += 0.1
                # 至多5轮就公布同长链
            else:
                self.Wait()
                self.tolerant = 0


        else:
            newblock, mine_success = self.Advermine()
            if self.mine_chain.lastblock.BlockHeight() == self.honest_chain.lastblock.BlockHeight():
                if random.uniform(0 + self.tolerant, 1) <= 0.8:
                    self.Wait()
                    # self.tolerant += 0.1
                    # 至多5轮就公布同长链
                else:
                    self.Match()
                    self.tolerant = 0
            if self.mine_chain.lastblock.BlockHeight() < self.honest_chain.lastblock.BlockHeight():
                if random.uniform(0 + self.tolerant, 1) <= 0.8:
                    self.Wait()
                    # self.tolerant += 0.1
                    # 至多5轮就公布同长链
                else:
                    self.Giveup()
                    self.tolerant = 0
            if self.mine_chain.lastblock.BlockHeight() > self.honest_chain.lastblock.BlockHeight():
                if random.uniform(0 + self.tolerant, 1) <= 0.2:
                    self.Override()
                    # self.tolerant += 0.1
                    # 至多5轮就公布同长链
                else:
                    self.Wait()
                    self.tolerant = 0

    def Normal_excute(self):

        honest_update = self.Adopt()
        #if honest_update:
            #print("更新诚实链")
        newblock, mine_success = self.Advermine()
        if self.mine_chain.lastblock.BlockHeight() > self.honest_chain.lastblock.BlockHeight():
            self.Override()
        else:
            if self.mine_chain.lastblock.BlockHeight() < self.honest_chain.lastblock.BlockHeight() - self.tolerant_len:
                self.Giveup()
            else:
                self.Wait()

    # @get_time
    def excute(self, curround):
        self.round = curround
        if self.excute_type == 'Normal':
            self.Normal_excute()
        if self.excute_type == 'Probability':
            self.Probability_excute()
        if self.round == self.num_rounds:
            self.attacklog2txt()
        # self.printchainlen()