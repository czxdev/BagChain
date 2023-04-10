import time
import math
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Attack import Attack, Selfmining
from functions import for_name
import global_var
from consensus import Consensus
from chain import Block, Chain
from miner import Miner
from task import Task
from external import V, I ,R , common_prefix, chain_quality, chain_growth, printchain2txt

logger = logging.getLogger(__name__)

def get_time(f):

    def inner(*arg,**kwarg):
        s_time = time.time()
        res = f(*arg,**kwarg)
        e_time = time.time()
        print('耗时：{}秒'.format(e_time - s_time))
        return res
    return inner

class Environment(object):
    '''包括了整个仿真器的一切参数和数据，驱动仿真器的其余要素完成仿真任务'''
    def __init__(self,  t, q, target,network_param,*adversary_ids):
        #environment parameters
        self.miner_num = global_var.get_miner_num()  # number of miners
        self.max_adversary = t  # maximum number of adversary
        self.qmax = q  # number of hash trials in a round
        self.target = target
        self.total_round = 0
        self.global_chain = Chain()  # a global tree-like data structure
        # 计划任务的队列schedule（用来发布数据集）
        # 队列中中每一项为一个三元元组(任务执行时间、任务执行函数、任务附加数据)
        self.schedule = []
        self.global_miniblock_list = [] # 记录所有产生过的miniblock
        self.dataset_publish_history = [] # 记录所有发布过的数据集
        self.winning_block_record = {} # {height : block.name}
        self.test_set_metric = {} # versus block height Key:Height Value:Metric
        self.validation_set_metric = {} # versus block height Key:Height Value:Metric
        # generate miners
        self.miners = []
        miner_i = 0
        for _ in range(self.miner_num):
            self.miners.append(Miner(miner_i, self.qmax, self.target))
            miner_i = miner_i + 1
        self.adversary_mem = []
        self.select_adversary(*adversary_ids)
        # generate network
        self.network = for_name(global_var.get_network_type())(self.miners)    #网络类型
        print(
            '\nParameters:','\n',
            'Miner Number: ', self.miner_num,'\n',
            'q: ', self.qmax, '\n',
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
            adversary.set_Adversary(True)
        return self.adversary_mem

    def select_adversary(self,*Miner_ID):
        '''
        根据指定ID选择对手
        return:self.adversary_mem
        '''   
        for miner in Miner_ID:
            self.adversary_mem.append(self.miners[miner])
            self.miners[miner].set_Adversary(True)
        return self.adversary_mem
    
    def clear_adversary(self):
        '''清空所有对手'''
        for adversary in self.adversary_mem:
            adversary.set_Adversary(False)
        self.adversary_mem=[]
        
    def exec(self, num_rounds):
        '''
        调用当前miner的BackboneProtocol完成mining
        当前miner用addblock功能添加上链
        之后gobal_chain用深拷贝的addchain上链
        '''
        if self.adversary_mem:
            attack = Selfmining(self.global_chain, self.target, self.network, self.adversary_mem, num_rounds)
        t_0 = time.time()
        network_idle_counter = 0 # 网络闲置的轮数
        for round in range(1, num_rounds+1):
            inputfromz = round
            temp_miniblock_list = [] # 临时存放miniblock
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
                    # Attack 不提供这个操作
                    # run the bitcoin backbone protocol
                    newblock = self.miners[i].BackboneProtocol(round)
                    if newblock is not None:
                        self.network.access_network(newblock,self.miners[i].Miner_ID,round)
                        network_idle_counter = 0
                        if newblock.blockextra.blocktype is newblock.BlockType.BLOCK: # 完整的区块要合并到全局链
                            self.global_chain.AddChain(newblock)
                        elif newblock not in self.global_miniblock_list:
                            temp_miniblock_list.append(newblock)

                    self.miners[i].input_tape = []  # clear the input tape
                    self.miners[i].receive_tape = []  # clear the communication tape
            last_winningblock = None
            self.global_miniblock_list.extend(temp_miniblock_list)
            for miniblock in temp_miniblock_list:
                preblock_height = miniblock.last.blockhead.height # 前一区块的高度
                self.validation_set_metric.setdefault(preblock_height, 0)
                self.winning_block_record.setdefault(preblock_height, miniblock.last.name)
                if self.validation_set_metric[preblock_height] < miniblock.blockextra.validate_metric or \
                   self.validation_set_metric[preblock_height] == miniblock.blockextra.validate_metric and \
                   global_var.get_miniblock_num(self.winning_block_record[preblock_height]) < \
                   global_var.get_miniblock_num(miniblock.last.name) or miniblock.last.isGenesis:
                    # 前一区块验证集性能更好
                    # 或者验证集性能相同但是miniblock更多,应该成为获胜区块
                    self.validation_set_metric[preblock_height] = miniblock.blockextra.validate_metric
                    # 确定上一高度的获胜区块
                    last_winningblock = miniblock.last
                    self.winning_block_record[preblock_height] = last_winningblock.name
                    self.test_set_metric[preblock_height] = last_winningblock.blockextra.metric

            if last_winningblock:
                for task in self.schedule:
                # 从schedule中删除同一高度、同一prehash的数据集发布任务
                    if task[1].blockhead.height == last_winningblock.blockhead.height and \
                       task[1].blockhead.prehash == last_winningblock.blockhead.prehash:
                        self.schedule.remove(task)

                # schedule dataset publication
                dataset_info = (last_winningblock.blockhead.blockhash,
                                id(last_winningblock.blockextra.task_queue[0]))
                test_set_publish_task = (round + global_var.get_test_set_interval(),
                                        last_winningblock,
                                        (Task.DatasetType.TEST_SET,*dataset_info))
                validation_set_publish_task = (round + \
                                              global_var.get_validation_set_interval(),
                                              last_winningblock,
                                              (Task.DatasetType.VALIDATION_SET,*dataset_info))
                self.schedule.append(test_set_publish_task)
                self.schedule.append(validation_set_publish_task)
                
            for index, task in enumerate(self.schedule):
                if task[0] == round:
                    dataset_info = self.schedule.pop(index)[2]
                    for miner in self.miners:
                        miner.dataset_publication_channel.append((*dataset_info, time.time_ns()))
                    logger.info("%s published prehash:%s taskid:%d at round %d",
                    dataset_info[0].name, *dataset_info[1:], round)

            # 错误检查，如果超过一定轮数没有新区块或miniblock，可能是系统出现未知错误
            network_idle_counter += 1 # 没有新的区块或miniblock，闲置轮数+1
            if network_idle_counter > 100: # 如果调整了区块与miniblock大小，注意修改该阈值
                logger.error("Blockchain system freeze, no more blocks or miniblocks")

            self.network.diffuse(round) 
            self.assess_common_prefix()
            self.process_bar(round, num_rounds, t_0) # 显示进度条
        
        # self.clear_adversary()
        self.total_round = self.total_round + num_rounds

    def assess_common_prefix(self):
        '''Assess Common Prefix Property'''
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
        '''展示一些仿真结果'''
        print('\n')
        for i,miner in enumerate(self.miners):
            printchain2txt(miner, ''.join(['chain_data', str(i), '.txt']))
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
        cq_dict, chain_quality_property = chain_quality(self.miners[0].Blockchain)
        print('Chain_Quality Property:', cq_dict)
        print('Ratio of blocks contributed by malicious players:', round(chain_quality_property, 3))
        print('Upper Bound t/(n-t):', round(self.max_adversary / (self.miner_num - self.max_adversary), 3))

        self.global_chain.ShowStructure(self.miner_num)

        # block interval distribution
        # self.miners[0].Blockchain.Get_block_interval_distribution()

        print('Visualizing Blockchain...')
        self.global_chain.ShowStructureWithGraphviz()
        self.global_chain.ShowStructureWithGraphvizWithAllMiniblock(self.global_miniblock_list)

        if self.network.__class__.__name__=='TopologyNetwork':
            pass # self.network.gen_routing_gragh_from_json() 由于生成路由图表占用大量时间空间，禁用
        # print_chain_property2txt(self.miners[9].Blockchain)
        test_set_metric_list = list(self.test_set_metric.values())[1:] # 排除创世区块
        average_test_set_metric = sum(test_set_metric_list)/len(test_set_metric_list)
        validation_set_metric_list = list(self.validation_set_metric.values())[1:] # 排除创世区块
        average_validation_set_metric_list = \
            sum(validation_set_metric_list)/len(validation_set_metric_list)
        print("Averaged metric:")
        print("Test Set:", average_test_set_metric)
        print("Validation Set:", average_validation_set_metric_list)
        logger.info("Averaged metric:\nTest Set: %f\nValidation Set: %f\n",
                    average_test_set_metric,
                    average_validation_set_metric_list)
        self.plot_metric_against_height()
        print("End")

    def plot_metric_against_height(self):
        '''绘制性能指标随高度的变化曲线'''
        fig, ax = plt.subplots()
        height_test = list(self.test_set_metric.keys())[1:]
        metric_test = list(self.test_set_metric.values())[1:]
        ax.plot(height_test,metric_test,'b^-',label="test set metric")
        height_validate = list(self.validation_set_metric.keys())[1:]
        metric_validate = list(self.validation_set_metric.values())[1:]
        ax.plot(height_validate,metric_validate,'r+-',label="validation set metric")
        ax.set_xlabel("height")
        ax.set_ylabel("metric")
        ax.set_title("Block metric vs Height")
        ax.legend()
        ax.grid(True)
        fig.savefig(global_var.get_result_path()/"metric_plot.svg")
        if global_var.get_show_fig():
            fig.show()
    
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
        print("\r{}{}  {:.5f}%  {}/{}  {:.2f} round/s  {}:{}:{}>>{}:{}:{}  Events: see events.log     "\
        .format(cplt, uncplt, percent*100, process, total, vel, time_cost.tm_hour, time_cost.tm_min, time_cost.tm_sec,\
            time_eval.tm_hour, time_eval.tm_min, time_eval.tm_sec),end="", flush=True)
