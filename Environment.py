import copy
import time
import math
import random
from typing import List

import numpy as np


import global_var
import network
from chain import Chain
from miner import Miner
from Attack import Selfmining
from functions import for_name
from external import common_prefix, chain_quality, chain_growth, printchain2txt



def get_time(f):

    def inner(*arg,**kwarg):
        s_time = time.time()
        res = f(*arg,**kwarg)
        e_time = time.time()
        print('耗时：{}秒'.format(e_time - s_time))
        return res
    return inner

class Environment(object):

    def __init__(self,  t = None, q_ave = None, q_distr = None, target = None, network_param = None, *adversary_ids, **genesis_blockextra):
        #environment parameters
        self.miner_num = global_var.get_miner_num()  # number of miners
        self.max_adversary = t  # maximum number of adversary
        self.q_ave = q_ave  # number of hash trials in a round
        self.q_distr = [] #
        self.target = target
        self.total_round = 0
        self.global_chain = Chain()  # a global tree-like data structure
        # generate miners
        self.miners:List[Miner] = []
        self.create_miners_q_rand() if q_distr =='rand' else self.create_miners_q_equal()
        print(genesis_blockextra)
        self.envir_create_genesis_block(**genesis_blockextra)
        self.adversary_mem:List[Miner] = []
        self.select_adversary(*adversary_ids)
        # generate network
        self.network:network.Network = for_name(global_var.get_network_type())(self.miners)    #网络类型
        print(
            '\nParameters:','\n',
            'Miner Number: ', self.miner_num,'\n',
            'q_ave: ', self.q_ave, '\n', 
            'Adversary Miners: ', adversary_ids, '\n',
            'Consensus Protocol: ', global_var.get_consensus_type(), '\n',
            'Target: ', self.target, '\n',
            'Network Type: ', self.network.__class__.__name__, '\n', 
            'Network Param: ', network_param, '\n'
        )
        self.network.set_net_param(**network_param)
        # evaluation
        self.selfblock = []
        self.max_suffix = 10
        self.cp_pdf = np.zeros((1, self.max_suffix)) # 每轮结束时，各个矿工的链与common prefix相差区块个数的分布

    def select_adversary_random(self):
        '''
        随机选择对手
        return:self.adversary_mem
        '''
        self.adversary_mem=random.sample(self.miners,self.max_adversary)
        for adversary in self.adversary_mem:
            adversary.set_adversary(True)
        return self.adversary_mem


    def select_adversary(self,*Miner_ID):
        '''
        根据指定ID选择对手
        return:self.adversary_mem
        '''   
        for miner in Miner_ID:
            self.adversary_mem.append(self.miners[miner])
            self.miners[miner].set_adversary(True)
        return self.adversary_mem
    
    def clear_adversary(self):
        '''清空所有对手'''
        for adversary in self.adversary_mem:
            adversary.set_adversary(False)
        self.adversary_mem=[]

    def create_miners_q_equal(self):
        for miner_id in range(self.miner_num):
            self.miners.append(Miner(miner_id, self.q_ave, self.target))

    def create_miners_q_rand(self):
        '''
        随机设置各个节点的hash rate,满足均值为q_ave,方差为1的高斯分布
        且满足全网总算力为q_ave*miner_num
        '''
        # 生成均值为ave_q，方差为1的高斯分布
        q_dist = np.random.normal(self.q_ave, self.miner_num)
        # 归一化到总和为total_q，并四舍五入为整数
        total_q = self.q_ave * self.miner_num
        q_dist = total_q / np.sum(q_dist) * q_dist
        q_dist = np.round(q_dist).astype(int)
        # 修正，如果和不为total_q就把差值分摊在最小值或最大值上
        if np.sum(q_dist) != total_q:
            diff = total_q - np.sum(q_dist)
            for _ in range(abs(diff)):
                sign_diff = np.sign(diff)
                idx = np.argmin(q_dist) if sign_diff > 0 else np.argmax(q_dist)
                q_dist[idx] += sign_diff
        for miner_id in range(self.miner_num):
            for q in q_dist:
                self.miners.append(Miner(miner_id, q, self.target))
        return q_dist
    

    def envir_create_genesis_block(self, **blockextra):
        '''create genesis block for all the miners in the system.'''
        self.global_chain.create_genesis_block(**blockextra)
        for miner in self.miners:
            miner.Blockchain.create_genesis_block(**blockextra)

        
    #@get_time
    def exec(self, num_rounds):

        '''
        调用当前miner的BackboneProtocol完成mining
        当前miner用addblock功能添加上链
        之后gobal_chain用深拷贝的addchain上链
        '''
        if self.adversary_mem:
            attack = Selfmining(self.global_chain, self.target, self.network, self.adversary_mem, num_rounds)
        t_0 = time.time()
        for round in range(1, num_rounds+1):
            #print("")
            #print("Round:{}".format(round))
            inputfromz = round
            # A随机选叛变
            # self.network.clear_NetworkTape()
            for attacker in self.adversary_mem:
                attacker.input_tape.append(("INSERT", inputfromz))
            if self.adversary_mem:
                attack.excute(round)
            # Adver的input_tape在excute里清空了
            for i in range(self.miner_num):
                #print("Miner{} is mining".format(i))
                if not self.miners[i].isAdversary:
                    self.miners[i].input_tape.append(("INSERT", inputfromz))
                    ''' Attack 不提供这个操作'''
                    # run the bitcoin backbone protocol
                    newblock = self.miners[i].BackboneProtocol(round)
                    if newblock is not None:
                        self.network.access_network(newblock,self.miners[i].Miner_ID,round)
                        self.global_chain.add_block_copy(newblock)
                    self.miners[i].input_tape = []  # clear the input tape
                    self.miners[i].receive_tape = []  # clear the communication tape
            self.network.diffuse(round)  # diffuse(C)
            self.assess_common_prefix()
            self.process_bar(round, num_rounds, t_0) # 这个是显示进度条的，如果不想显示，注释掉就可以
            # 分割一下
        # self.clear_adversary()
        self.total_round = self.total_round + num_rounds

    def assess_common_prefix(self):
        # Common Prefix Property
        cp = self.miners[0].Blockchain.lastblock
        for i in range(1, self.miner_num):
            if not self.miners[i].isAdversary:
                cp = common_prefix(cp, self.miners[i].Blockchain)
        len_cp = cp.blockhead.height
        for i in range(0, self.miner_num):
            len_suffix = self.miners[0].Blockchain.lastblock.blockhead.height - len_cp
            if len_suffix >= 0 and len_suffix < self.max_suffix:
                self.cp_pdf[0, len_suffix] = self.cp_pdf[0, len_suffix] + 1

    def view(self):
        # 展示一些仿真结果
        print('\n')
        miner_i = 0
        while miner_i < self.miner_num:
            # print("Blockchain of Miner", miner_i, ":", "")
            # self.miners[miner_i].Blockchain.ShowLChain()
            printchain2txt(self.miners[miner_i], ''.join(['chain_data', str(miner_i), '.txt']))
            miner_i = miner_i + 1
        print("Global Tree Structure:", "")
        self.global_chain.ShowStructure1()
        print("End of Global Tree", "")
        # self.miners[9].ValiChain()

        # Chain Growth Property
        print('Chain Growth Property:')
        growth = 0
        num_honest = 0
        for i in range(self.miner_num):
            if not self.miners[i].isAdversary:
                growth = growth + chain_growth(self.miners[i].Blockchain)
                num_honest = num_honest + 1
        growth = growth / num_honest
        stats = self.global_chain.CalculateStatistics(self.total_round)
        print(stats["num_of_generated_blocks"], "blocks are generated in",
              self.total_round, "rounds, in which", stats["num_of_stale_blocks"], "are stale blocks.")
        print("Average chain growth in honest miners' chain:", round(growth, 3))
        print("Number of Forks:", stats["num_of_forks"])
        print("Fork rate:", round(stats["fork_rate"], 3))
        print("Stale rate:", round(stats["stale_rate"], 3))
        print("Average block time (main chain):", round(stats["average_block_time_main"]), "rounds/block")
        print("Block throughput (main chain):", round(stats["block_throughput_main"],3), "blocks/round")
        print("Throughput in MB (main chain):", round(stats["throughput_main_MB"], 3), "blocks/round")
        print("Average block time (total):", round(stats["average_block_time_total"]), "rounds/block")
        print("Block throughput (total):", round(stats["block_throughput_total"], 3), "blocks/round")
        print("Throughput in MB (total):", round(stats["throughput_total_MB"], 3), "MB/round")
        print("")

        # Common Prefix Property
        print('Common Prefix Property:')
        print('The common prefix pdf:')
        print(self.cp_pdf)
        print('Consistency rate:',self.cp_pdf[0,0]/(self.cp_pdf.sum()))
        print("")

        # Chain Quality Property
        cq_dict, chain_quality_property = chain_quality(self.miners[9].Blockchain)
        print('Chain_Quality Property:', cq_dict)
        print('Ratio of blocks contributed by malicious players:', round(chain_quality_property, 3))
        print('Upper Bound t/(n-t):', round(self.max_adversary / (self.miner_num - self.max_adversary), 3))

        self.global_chain.ShowStructure(self.miner_num)

        # block interval distribution
        # self.miners[0].Blockchain.Get_block_interval_distribution()

        self.global_chain.ShowStructureWithGraphviz()

        ave_block_propagation_times = self.network.cal_block_propagation_times()
        print('Block propagation times:', ave_block_propagation_times)


        if self.network.__class__.__name__=='TopologyNetwork':
            self.network.gen_routing_gragh_from_json()
        # print_chain_property2txt(self.miners[9].Blockchain)

    def showselfblock(self):
        print("")
        print("Adversary的块：")
        for block in self.selfblock:
            print(block.name)

    def process_bar(self,process,total,t_0):
        bar_len = 50
        percent = (process)/total
        cplt = "■" * math.ceil(percent*bar_len)
        uncplt = "□" * (bar_len - math.ceil(percent*bar_len))
        time_len = time.time()-t_0+0.0000000001
        time_cost = time.gmtime(time_len)
        vel = process/(time_len)
        time_eval = time.gmtime(total/(vel+0.001))
        print("\r{}{}  {:.5f}%  {}/{}  {:.2f} round/s  {}:{}:{}>>{}:{}:{}  Events: see events.log "\
        .format(cplt, uncplt, percent*100, process, total, vel, time_cost.tm_hour, time_cost.tm_min, time_cost.tm_sec,\
            time_eval.tm_hour, time_eval.tm_min, time_eval.tm_sec),end="", flush=True)
