import time
import math
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import copy
import time
import math
import random
from typing import List
from collections import defaultdict
import numpy as np


import global_var
import network
from chain import Block, Chain, BlockList
from miner import Miner
from Attack import Selfmining
from functions import for_name
from external import common_prefix, chain_quality, chain_growth, printchain2txt
from task import Task
from tasks.models import NNClassifier

logger = logging.getLogger(__name__)

def get_time(f):

    def inner(*arg,**kwarg):
        s_time = time.time()
        res = f(*arg,**kwarg)
        e_time = time.time()
        print('耗时：{}秒'.format(e_time - s_time))
        return res
    return inner

def stash_models(miniblock:Block):
    '''stash the NN models to RAM in the referenced Miniblocks'''
    if isinstance(miniblock.blockextra.model, list) and \
        len(miniblock.blockextra.model) > 0 and \
        isinstance(miniblock.blockextra.model[0], NNClassifier):
        for model in miniblock.blockextra.model:
            model.net.cpu()
    elif isinstance(miniblock.blockextra.model, NNClassifier):
        miniblock.blockextra.model.net.cpu()

class Environment(object):

    def __init__(self,  t = None, q_ave = None, q_distr = None, target = None, network_param = None, *adversary_ids, **genesis_blockextra):
        '''包括了整个仿真器的一切参数和数据，驱动仿真器的其余要素完成仿真任务'''
        #environment parameters
        self.miner_num = global_var.get_miner_num()  # number of miners
        self.max_adversary = t  # maximum number of adversary
        self.q_ave = q_ave  # number of hash trials in a round
        self.q_distr = [] #
        self.target = target
        self.total_round = 0
        self.global_chain = Chain()  # a global tree-like data structure
        # generate miners
        self.miners:list[Miner] = []
        self.create_miners_q_rand() if q_distr =='rand' else self.create_miners_q_equal()
        print(genesis_blockextra)
        self.envir_create_genesis_block(**genesis_blockextra)
        self.adversary_mem:List[Miner] = []
        self.select_adversary(*adversary_ids)
        # 计划任务的队列schedule（用来发布数据集）
        # 队列中中每一项为一个三元元组(任务执行时间、任务执行函数、任务附加数据)
        self.schedule = []
        self.schedule_initial_tasks()
        self.global_miniblock_list = [] # 记录所有产生过的miniblock
        self.global_ensemble_block_list = [] # 记录所有已产生的ensemble block
        self.dataset_publish_history = [] # 记录所有发布过的数据集
        self.winning_block_record = {} # {height : block.name}
        self.test_set_metric = {} # versus block height Key:Height Value:Metric
        self.validation_set_metric = {} # versus block height Key:Height Value:Metric
        self.base_model_metric_lists = {} # versus block height Key:Height Value:(list[Metric], list[Metric])
                                        #   test and validation list of the base models in a winning key block
        self.dummy_consensus = for_name(global_var.get_consensus_type())(-1)
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
        self.block_checkpoints = [self.global_chain.lastblock]

    def schedule_initial_tasks(self):
        '''初始化计划任务队列schedule'''
        genesis_block = self.global_chain.head
        dataset_info = (genesis_block.blockhead.blockhash,
                        id(genesis_block.blockextra.task_queue[0]),
                        genesis_block.blockhead.height)
        test_set_publish_task = (global_var.get_test_set_interval(),
                                genesis_block,
                                (Task.DatasetType.TEST_SET,*dataset_info))
        validation_set_publish_task = (global_var.get_test_set_interval() + \
                                       global_var.get_validation_set_interval(),
                                        genesis_block,
                                        (Task.DatasetType.VALIDATION_SET,*dataset_info))
        self.schedule.append(test_set_publish_task)
        self.schedule.append(validation_set_publish_task)

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
    
    def handle_block(self, newblock:Block, miner_id, round):
        '''处理BackboneProtocol返回的Block对象
        param:
            newblock 待处理的Block对象 type:Block
            miner_id 矿工ID
            round 当前轮次
        return:
            blocktype Block对象类型，如果没有返回有效块则返回
        '''
        if newblock is None:
            return None
        self.network.access_network(newblock,miner_id,round)
        if newblock.blockextra.blocktype is newblock.BlockType.KEY_BLOCK: # Key Block要合并到全局链
            self.global_chain.add_block_copy(newblock)
        elif newblock.blockextra.blocktype is newblock.BlockType.MINIBLOCK and \
            newblock not in self.global_miniblock_list:
            self.global_miniblock_list.append(newblock)
        elif newblock.blockextra.blocktype is newblock.BlockType.BLOCK and \
            newblock not in self.global_ensemble_block_list:
            self.global_ensemble_block_list.append(newblock)

        return newblock.blockextra.blocktype

    def envir_create_genesis_block(self, **blockextra):
        '''create genesis block for all the miners in the system.'''
        self.global_chain.create_genesis_block(**blockextra)
        for miner in self.miners:
            miner.Blockchain.create_genesis_block(**blockextra)

    #@get_time
    def exec(self, num_rounds, max_height):

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
            temp_key_block_list:BlockList = [] # 临时存放当前轮次生成的Key Block
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
                    new_block = self.miners[i].BackboneProtocol(round)
                    if new_block is not None:
                        network_idle_counter = 0
                    if self.handle_block(new_block, self.miners[i].Miner_ID, round) \
                        is Block.BlockType.KEY_BLOCK:
                        temp_key_block_list.append(new_block)
                    self.miners[i].input_tape = []  # clear the input tape
                    self.miners[i].receive_tape = []  # clear the communication tape
            last_winningblock = None

            for key_block in temp_key_block_list:
                preblock_height = key_block.blockhead.height # 获胜Key Block所在的高度
                self.validation_set_metric.setdefault(preblock_height, 0)
                self.winning_block_record.setdefault(preblock_height, key_block.name)
                if self.validation_set_metric[preblock_height] < key_block.blockextra.metric:
                    # 该Key Block验证集性能更好
                    height_list = [dataset_info[3] for dataset_info in self.dataset_publish_history]
                    if preblock_height in height_list:
                        # 如果当前高度的测试集已经发布就不再处理当前高度上的Key Block
                        continue
                    last_winningblock = key_block
                    self.validation_set_metric[preblock_height] = last_winningblock.blockextra.metric
                    self.winning_block_record[preblock_height] = last_winningblock.name
                    winning_ensemble_block_hashs = [k for k,v in last_winningblock.blockextra.validate_list.items() \
                             if v == last_winningblock.blockextra.metric]
                    winning_ensemble_block = None
                    for ensemble_block in last_winningblock.blockextra.ensemble_block_list:
                        if ensemble_block.blockhead.blockhash in winning_ensemble_block_hashs:
                            winning_ensemble_block = ensemble_block
                            break # 匹配到就直接退出循环（不全部匹配）
                    self.test_set_metric[preblock_height] = winning_ensemble_block.blockextra.metric
                    base_metrics_test = np.nan * np.ones(self.miner_num)
                    base_metrics_validation = np.nan * np.ones(self.miner_num)
                    for miniblock in winning_ensemble_block.blockextra.miniblock_list:
                        model_id = miniblock.blockhead.miner
                        base_metrics_test[model_id] = self.dummy_consensus.validate_evaluate_miniblock([miniblock], global_var.get_global_task(), Task.DatasetType.TEST_SET)
                        base_metrics_validation[model_id] = self.dummy_consensus.validate_evaluate_miniblock([miniblock], global_var.get_global_task(), Task.DatasetType.VALIDATION_SET)
                    self.base_model_metric_lists[preblock_height] = (base_metrics_test, base_metrics_validation)

            if last_winningblock:
                # 从schedule中删除同一高度的数据集发布任务
                self.schedule = [task for task in self.schedule if \
                                 task[1].blockhead.height != last_winningblock.blockhead.height]

                # schedule dataset publication
                dataset_info = (last_winningblock.blockhead.blockhash,
                                id(last_winningblock.blockextra.task_queue[0]),
                                last_winningblock.blockhead.height)
                test_set_publish_task = (round + global_var.get_test_set_interval(),
                                        last_winningblock,
                                        (Task.DatasetType.TEST_SET,*dataset_info))
                validation_set_publish_task = (round + global_var.get_test_set_interval() +\
                                              global_var.get_validation_set_interval(),
                                              last_winningblock,
                                              (Task.DatasetType.VALIDATION_SET,*dataset_info))
                self.schedule.append(test_set_publish_task)
                self.schedule.append(validation_set_publish_task)

            new_schedule = []
            for task in self.schedule:
                if task[0] != round:
                    new_schedule.append(task)
                else:
                    dataset_info = task[2]
                    for miner in self.miners:
                        miner.dataset_publication_channel.append((*dataset_info, time.time_ns()))
                    logger.info("%s published prehash:%s taskid:%d height:%d at round %d",
                    dataset_info[0].name, *dataset_info[1:4], round)
                    self.dataset_publish_history.append(dataset_info)
            self.schedule = new_schedule

            # 四个区块高度之后，查找Key Block中的MiniBlock并将模型移动到RAM中节省VRAM
            checkpoint_height = self.block_checkpoints[0].blockhead.height
            if self.global_chain.lastblock.blockhead.height - checkpoint_height >= 4:
                new_checkpoints = []
                for block in self.block_checkpoints:
                    new_checkpoints.extend(block.next)
                stashed_miniblock_names = []
                for miniblock in self.global_miniblock_list:
                    if miniblock.blockhead.height > checkpoint_height + 1:
                        break
                    if miniblock.blockhead.height == checkpoint_height:
                        stash_models(miniblock)
                        stashed_miniblock_names.append(miniblock.name)
                logger.info(f"Stashed miniblocks on height {checkpoint_height}: {stashed_miniblock_names}")

                self.block_checkpoints = new_checkpoints

            if self.global_chain.lastblock.BlockHeight() > max_height:
                self.total_round = self.total_round + round
                return

            # 错误检查，如果超过一定轮数没有新区块或miniblock，可能是系统出现未知错误
            network_idle_counter += 1 # 没有新的block，闲置轮数+1
            if network_idle_counter > 1000: # 如果调整了区块与miniblock大小，注意修改该阈值
                logger.warning("Blockchain system freeze, no more blocks or miniblocks")

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

        # Evaluation Results
        stats = self.global_chain.CalculateStatistics(self.total_round)
        # Chain Growth Property
        growth = 0
        num_honest = 0
        for i in range(self.miner_num):
            if not self.miners[i].isAdversary:
                growth = growth + chain_growth(self.miners[i].Blockchain)
                num_honest = num_honest + 1
        growth = growth / num_honest
        stats.update({
            'average_chain_growth_in_honest_miners\'_chain': growth
        })
        # Common Prefix Property
        stats.update({
            'common_prefix_pdf': self.cp_pdf,
            'consistency_rate':self.cp_pdf[0,0]/(self.cp_pdf.sum())
        })
        # Chain Quality Property
        cq_dict, chain_quality_property = chain_quality(self.global_chain)
        stats.update({
            'chain_quality_property': cq_dict,
            'ratio_of_blocks_contributed_by_malicious_players': round(chain_quality_property, 3),
            'upper_bound t/(n-t)': round(self.max_adversary / (self.miner_num - self.max_adversary), 3)
        })
        # Network Property
        stats.update({'block_propagation_times': {} })
        if self.network.__class__.__name__ != 'SynchronousNetwork':
            ave_block_propagation_times = self.network.cal_block_propagation_times()
            stats.update({
                'block_propagation_times': ave_block_propagation_times
            })
        
        for k,v in stats.items():
            if type(v) is float:
                stats.update({k:round(v,3)})

        # save the results in the evaluation results.txt
        RESULT_PATH = global_var.get_result_path()
        with open(RESULT_PATH / 'evaluation results.txt', 'a+',  encoding='utf-8') as f:
            blocks_round = ['block_throughput_main', 'block_throughput_total']
            MB_round = ['throughput_main_MB', 'throughput_total_MB']
            rounds_block = ['average_block_time_main', 'average_block_time_total']

            for k,v in stats.items():
                if k in blocks_round:
                    print(f'{k}: {v} blocks/round', file=f)
                elif k in MB_round:
                    print(f'{k}: {v} MB/round', file=f)
                elif k in rounds_block:
                    print(f'{k}: {v} rounds/block', file=f)
                else:
                    print(f'{k}: {v}', file=f)

        # show the results in the terminal
        # Chain Growth Property
        print('Chain Growth Property:')
        print(stats["num_of_generated_blocks"], "blocks are generated in",
              self.total_round, "rounds, in which", stats["num_of_stale_blocks"], "are stale blocks.")
        logger.info("%d blocks are generated in %d rounds, in which %d are stale blocks",
                    stats["num_of_generated_blocks"],self.total_round,stats["num_of_stale_blocks"])
        print("Average chain growth in honest miners' chain:", round(growth, 3))
        print("Number of Forks:", stats["num_of_forks"])
        print("Fork rate:", stats["fork_rate"])
        print("Stale rate:", stats["stale_rate"])
        logger.info("Average Local Blockchain Height in honest miners:%d", growth)
        logger.info("Fork rate:%f ,Slate rate:%f", round(stats["fork_rate"],3), round(stats["stale_rate"],3))
        print("Average block time (main chain):", stats["average_block_time_main"], "rounds/block")
        print("Block throughput (main chain):", stats["block_throughput_main"], "blocks/round")
        print("Throughput in MB (main chain):", stats["throughput_main_MB"], "blocks/round")
        print("Average block time (total):", stats["average_block_time_total"], "rounds/block")
        print("Block throughput (total):", stats["block_throughput_total"], "blocks/round")
        print("Throughput in MB (total):", stats["throughput_total_MB"], "MB/round")
        print("")
        # Common Prefix Property
        print('Common Prefix Property:')
        print('The common prefix pdf:')
        print(self.cp_pdf)
        print('Consistency rate:',self.cp_pdf[0,0]/(self.cp_pdf.sum()))
        print("")
        # Chain Quality Property
        print('Chain_Quality Property:', cq_dict)
        print('Ratio of blocks contributed by malicious players:', chain_quality_property)
        print('Upper Bound t/(n-t):', self.max_adversary / (self.miner_num - self.max_adversary))
        # Network Property
        print('Block propagation times:', ave_block_propagation_times)

        # show or save figures
        self.global_chain.ShowStructure(self.miner_num)
        # block interval distribution
        self.miners[0].Blockchain.get_block_interval_distribution()

        print('Visualizing Blockchain...')
        self.global_chain.ShowStructureWithGraphviz()
        self.global_chain.ShowStructureWithGraphvizWithEverything(self.global_miniblock_list,
                                                                  self.global_ensemble_block_list)

        network_stats = self.network.calculate_stats()
        print('Average network delay:',network_stats["average_network_delay"],'rounds')


        if self.network.__class__.__name__=='TopologyNetwork':
            pass # self.network.gen_routing_gragh_from_json() 由于生成路由图表占用大量时间空间，禁用
        # print_chain_property2txt(self.miners[9].Blockchain)
        test_set_metric_list = list(self.test_set_metric.values())
        average_test_set_metric = sum(test_set_metric_list)/len(test_set_metric_list)
        validation_set_metric_list = list(self.validation_set_metric.values())
        average_validation_set_metric_list = \
            sum(validation_set_metric_list)/len(validation_set_metric_list)
        print("Averaged metric:")
        print("Test Set:", average_test_set_metric)
        print("Validation Set:", average_validation_set_metric_list)
        logger.info("Averaged metric:\nTest Set: %f\nValidation Set: %f\n",
                    average_test_set_metric,
                    average_validation_set_metric_list)
        self.plot_metric_against_height()
        # average the metrics in base_metrics_list with regard to the height
        base_metrics_test_list = np.array([test for h, (test, _) in self.base_model_metric_lists.items()])
        base_metrics_validation_list = np.array([validation for h, (_, validation) in self.base_model_metric_lists.items()])
        average_base_metrics_test = np.nanmean(base_metrics_test_list, axis=0) # ignore nan
        average_base_metrics_validation = np.nanmean(base_metrics_validation_list, axis=0)

        result_collection = {'test_metric_average': average_test_set_metric,
                             'validation_metric_average': average_validation_set_metric_list,
                             'test_metric_list': test_set_metric_list,
                             'validation_metric_list': validation_set_metric_list,
                             'chain_stats': stats,
                             'base_model_metric_test': average_base_metrics_test.tolist(),
                             'base_model_metric_validation': average_base_metrics_validation.tolist()}

        if global_var.get_global_task().global_dataset is not None:
            # simulate the situation where only the global model is used for model training
            global_task = global_var.get_global_task()
            BaseModel = global_task.model_constructor()
            BaseModel.fit(*global_task.global_dataset)
            def evaluate_model(model, dataset):
                return global_task.metric_evaluator(model.predict(dataset[0]), dataset[1])
            test_metrics_on_global_dataset = evaluate_model(BaseModel, global_task.test_set)
            validation_metrics_on_global_dataset = evaluate_model(BaseModel, global_task.validation_set)
            result_collection.update({'test_metric_on_global_dataset': test_metrics_on_global_dataset,
                                      'validation_metric_on_global_dataset': validation_metrics_on_global_dataset})

        generated_miniblocks_per_height = defaultdict(list)
        valid_miniblocks_per_height = {}
        for miniblock in self.global_miniblock_list:
            generated_miniblocks_per_height[miniblock.blockhead.height].append(miniblock)
        
        for keyblock in iter(self.global_chain):
            if keyblock.isGenesis:
                continue
            # find the ensemble blocks with the best metric
            winning_ensemble_block_hashs = [k for k,v in keyblock.blockextra.validate_list.items() \
                             if v == keyblock.blockextra.metric]
            for ensemble_block in keyblock.blockextra.ensemble_block_list:
                
                if ensemble_block.blockhead.blockhash in winning_ensemble_block_hashs:
                    winning_ensemble_block = ensemble_block
                    break
            valid_miniblocks_per_height[keyblock.blockhead.height] = winning_ensemble_block.blockextra.miniblock_list

        average_generated_miniblocks_per_height = sum([len(generated_miniblocks_per_height[height]) for height in valid_miniblocks_per_height])/len(valid_miniblocks_per_height)
        average_valid_miniblocks_per_height = sum([len(miniblocks) for miniblocks in valid_miniblocks_per_height.values()])/len(valid_miniblocks_per_height)

        accuracy_upper_bound_for_all_generated_miniblocks = [] # First column: test set, second column: validation set
        for height in valid_miniblocks_per_height:
            miniblocks = generated_miniblocks_per_height[height]
            test_accuracy = self.dummy_consensus.validate_evaluate_miniblock(miniblocks, global_var.get_global_task(), Task.DatasetType.TEST_SET)
            validation_accuracy = self.dummy_consensus.validate_evaluate_miniblock(miniblocks, global_var.get_global_task(), Task.DatasetType.VALIDATION_SET)
            accuracy_upper_bound_for_all_generated_miniblocks.append([test_accuracy, validation_accuracy])
        
        # average_accuracy_upper_bound_for_all_generated_miniblocks = [test_set_accuracy, validation_set_accuracy]
        average_accuracy_upper_bound_for_all_generated_miniblocks = np.mean(accuracy_upper_bound_for_all_generated_miniblocks, axis=0)

        result_collection.update({'average_generated_miniblocks_per_height': average_generated_miniblocks_per_height,
                                  'average_valid_miniblocks_per_height': average_valid_miniblocks_per_height,
                                  'average_accuracy_upper_bound_for_all_generated_miniblocks': average_accuracy_upper_bound_for_all_generated_miniblocks.tolist()})

        return result_collection

    def plot_metric_against_height(self):
        '''绘制性能指标随高度的变化曲线'''
        plt.rc("font",family="Times New Roman")# 配置字体
        from matplotlib import rcParams
        config = {
        "font.family": 'serif', # 衬线字体
        "font.size": 10.5, # 相当于五号大小
        "font.serif": ['SimSun'] # 宋体
        }
        rcParams.update(config)
        fig, ax = plt.subplots()
        height_test = list(self.test_set_metric.keys())
        metric_test = list(self.test_set_metric.values())
        ax.plot(height_test,metric_test,'^-',label="Test Set",color=plt.get_cmap("RdBu")(0.85))
        height_validate = list(self.validation_set_metric.keys())
        metric_validate = list(self.validation_set_metric.values())
        ax.plot(height_validate,metric_validate,'+-',label="Validation Set",color=plt.get_cmap("RdBu")(0.15))
        ax.set_xlabel("Height")
        ax.set_ylabel("Accuracy")
        # ax.set_title("Block metric vs Height")
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
        print("\r{}{}  {:.5f}%  {}/{}  {:.2f} round/s  {}:{}:{}>>{}:{}:{}  Events: see events.log "\
        .format(cplt, uncplt, percent*100, process, total, vel, time_cost.tm_hour, time_cost.tm_min, time_cost.tm_sec,\
            time_eval.tm_hour, time_eval.tm_min, time_eval.tm_sec),end="", flush=True)
