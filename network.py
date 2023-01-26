"""network module 实现网络功能"""
import json
import os
import random
import sys
from abc import ABCMeta, abstractmethod
from math import ceil

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

import global_var
from chain import Block, BlockHead
from errorclasses import NetAdjError, NetMinerNumError


class Network(metaclass=ABCMeta):
    """网络抽象基类"""

    def __init__(self) -> None:
        self.MINER_NUM = global_var.get_miner_num()  # 网络中的矿工总数，为常量
        self.NET_RESULT_PATH = global_var.get_net_result_path()

    @abstractmethod
    def set_net_param(self, *args, **kargs):
        pass

    @abstractmethod
    def access_network(self, newblock, minerid, ruond):
        pass

    @abstractmethod
    def diffuse(self, round):
        pass


class FullConnectedNetwork(Network):
    """全连接无延迟广播"""

    def __init__(self, miners: list):
        super().__init__()
        self.miners = miners
        self.network_tape = []  # network_tape存储要广播的新块和对应的信息
        # 元素chain_packet为字典存储新块及'minerid'、'TTL'等信息

        with open(self.NET_RESULT_PATH + '\\' + 'network_log.txt', 'a') as f:
            print('Network Type: FullConnectedNetwork', file=f)

    def set_net_param(self):
        pass

    def access_network(self, newblock, minerid, round):
        """ 本轮新产生的块添加到network_tape\n
        param: newblock type:block
               minerid type:int
        """
        block_packet = {
            'minerid': minerid,
            'block': newblock
        }
        self.network_tape.append(block_packet)

    def clear_NetworkTape(self):
        """清空network_tape"""
        self.network_tape = []

    def diffuse(self, round):
        """全连接无延迟广播network_tape中的块\n
        param: miners 全部矿工 type:miner
        """
        if self.network_tape:
            for j in range(self.MINER_NUM):
                for block_packet in self.network_tape:
                    if j != block_packet['minerid']:  # network_tape添加到miner的receive_tape中
                        self.miners[j].receive_tape.append(block_packet['block'])
            self.clear_NetworkTape()

'''
class BoundedDelayNetwork(Network):
    """全连接BoundedDelay广播"""

    def __init__(self, miners: list):
        super().__init__()
        self.miners = miners
        self.miner_delaylist = []  # 各miner的网络delay，元素int:经过多少轮才能被其他miner接收到,
        self.network_tape = []
        self.genMinerDelay(5)  # 最大网络时延为5

        with open(self.NET_RESULT_PATH + '\\' + 'network_log.txt', 'a') as f:
            print('Network Type: BoundedDelayNetwork', file=f)

    def set_net_param(self):
        pass

    def genMinerDelay(self, maxdelay):
        """对每个矿工随机生成网络时延\n
        param: maxdelay 最大网络时延   type:int
        """
        for _ in range(self.MINER_NUM):
            self.miner_delaylist.append(random.randint(1, maxdelay))

    def access_network(self, newblock, minerid, round):
        """本轮新产生的链添加到network_tape\n
        param: newblock type:block
               minerid type:int
        """
        block_packet = {
            'minerid': minerid,
            'delay': self.miner_delaylist[minerid],
            'block': newblock
        }
        self.network_tape.append(block_packet)

    def diffuse(self, round):
        """Bounded-Delay广播
        """
        diffuse_index = []  # 可diffuse的下标索引
        if self.network_tape:
            diffuse_index = []  # 可diffuse的下标索引
            for i, block_packet in enumerate(self.network_tape):
                if block_packet['delay'] == 0:  # 广播delay值为0的链
                    diffuse_index.append(i)
                    for j in range(self.MINER_NUM):
                        if j != block_packet['minerid']:  # network_tape添加到miner的receive_tape中
                            self.miners[j].receive_tape.append(block_packet['block'])
                else:  # delay值不为0的链的delay减1
                    block_packet['delay'] = block_packet['delay'] - 1
        # 删除network_tape和chain_delaylist中已经传播完成的
        self.network_tape = [n for i, n in enumerate(self.network_tape) if i not in diffuse_index]
'''


class BoundedDelayNetwork(Network):
    """矿工以概率接收到区块，在特定轮数前必定所有矿工都收到区块"""

    def __init__(self, miners: list):
        super().__init__()
        self.miners = miners
        self.network_tape = []
        self.rcvprob_start = 0.25
        self.rcvprob_inc = 0.25
        # 结果打印准备
        with open(self.NET_RESULT_PATH + '\\' + 'network_log.txt', 'a') as f:
            print('Network Type: BoundedDelayNetwork', file=f)


    def set_net_param(self, rcvprob_start, rcvprob_inc):
        """设置网络参数\n
        param:  rcvprob_start:每个包进入网络时的接收概率,默认0.25
                rcvprob_inc:之后每轮增加的接收概率,默认0.25"""
        self.rcvprob_start = rcvprob_start
        self.rcvprob_inc = rcvprob_inc
        with open(self.NET_RESULT_PATH + '\\' + 'network_log.txt', 'a') as f:
            print('rcvprob_start:{},rcvprob_inc={}'.format(self.rcvprob_start, self.rcvprob_inc), file=f)


    def IsRecieved(self, rcvprob_th):
        """节点收到区块的概率
        param:rcvprob_th 接收到概率;
        """
        return random.uniform(0, 1) < rcvprob_th

    def access_network(self, newblock, minerid, round):
        """新块及信息打包添加到network_tape\n
        param: newblock type:block
               minerid type:int

        """
        if not self.miners[minerid].isAdversary:
            block_packet = {
                'minerid': minerid,
                'block': newblock,
                'received_miner': [minerid],  # 记录接收到该区块的矿工节点
                'received_round': [round],
                'recieve_prob': self.rcvprob_start  # 每次加self.rcvprob_inc
            }
            self.network_tape.append(block_packet)

        if self.miners[minerid].isAdversary:
            adversarylist = [minerid]
            adversaryroundlist = [round]
            for miner in self.miners:
                if miner.isAdversary and miner.Miner_ID != minerid:
                    adversarylist.append(miner.Miner_ID)
                    adversaryroundlist.append(round)
                    miner.receiveBlock(newblock)

            block_packet = {
                'minerid': minerid,
                'block': newblock,
                'received_miner': adversarylist,  # 记录接收到该区块的矿工节点
                'received_round': adversaryroundlist,
                'recieve_prob': self.rcvprob_start  # 每次加self.rcvprob_inc
            }
            self.network_tape.append(block_packet)


    def diffuse(self, round):
        """ReceiveProb广播"""
        # recieve_prob=0.7#设置接收概率，目前所有矿工概率一致

        # 随着轮数的增加，收到的概率越高，无限轮
        # 超过某轮后所有人收到

        # 一个人收到之后就不会再次收到这个块了

        with open(self.NET_RESULT_PATH + '\\' + 'network_log.txt', 'a') as f:

            if self.network_tape:
                died_packets = []  # TTL=0的包的索引
                for i, block_packet in enumerate(self.network_tape):
                    for j in range(self.MINER_NUM):
                        if j != block_packet['minerid'] and j not in block_packet['received_miner']:
                            # 不会重复传给某个矿工
                            if self.IsRecieved(block_packet['recieve_prob']):
                                if not self.miners[j].isAdversary:
                                    # print(self.IsRecieved(block_packet['recieve_prob']))
                                    block_packet['received_miner'].append(self.miners[j].Miner_ID)
                                    block_packet['received_round'].append(round)
                                    self.miners[j].receiveBlock(block_packet['block'])
                                    # print(block_packet['block'].name,block_packet['recieve_prob'],block_packet['received_miner'],file=f)
                                    # 如果接收到了区块，则通过receiveBlock函数传递给该miner
                                    # if not miners[j].receiveBlock(block_packet['block']):#判断最近是否收到过该区块

                                # 判断如果是isAdversary:
                                # 如果一个ad收到，其他ad也立即收到
                                if self.miners[j].isAdversary:
                                    block_packet['received_miner'].append(self.miners[j].Miner_ID)
                                    block_packet['received_round'].append(round)
                                    self.miners[j].receiveBlock(block_packet['block'])
                                    for miner in self.miners:
                                        if miner.isAdversary and miner.Miner_ID != j:
                                            block_packet['received_miner'].append(miner.Miner_ID)
                                            block_packet['received_round'].append(round)
                                            miner.receiveBlock(block_packet['block'])

                    if block_packet['recieve_prob'] < 1:
                        block_packet['recieve_prob'] = block_packet['recieve_prob'] + self.rcvprob_inc  # 每轮
                    # block_packet['TTL']=block_packet['TTL']-1#TTL-1
                    if len(set(block_packet['received_miner'])) == self.MINER_NUM:  # 如果所有人都收到了，就丢弃该包
                        died_packets.append(i)
                        # 打印将要被丢弃的包的信息
                        print(block_packet['block'].name, 'is recieved by miners:', block_packet['received_miner'],
                              'in rounds:', block_packet['received_round'], file=f)
                # 丢弃TTL=0的包，更新network_tape
                self.network_tape = [n for i, n in enumerate(self.network_tape) if i not in died_packets]
                died_packets = []


class TopologyNetwork(Network):
    """拓扑P2P网络"""

    class BlockPacketTpNet(object):
        """嵌套类
        #拓扑网络中的区块数据包，包含路由相关信息"""
        def __init__(self, newblock: Block, minerid, round, TTL, outnetobj):
            self.block = newblock
            self.minerid = minerid
            self.round = round
            self.TTL = TTL  # 剩余存活时间
            self.outnetobj = outnetobj  # 外部网络类实例
            # 路由过程相关
            self.received_miners = [minerid]
            self.next_miners_and_delays = [[minerid, mi, d] for mi, d in
                                           zip(self.outnetobj.miners[minerid].neighbor_list,
                                               self.outnetobj.cal_neighbor_delays(newblock, minerid))]
            # 路由结果记录相关
            self.routing_histroy = {(minerid, target): [round, 0] for target in
                                    self.outnetobj.miners[minerid].neighbor_list}

                                    

    def __init__(self, miners: list):
        super().__init__()
        self.miners = miners
        self.TTL = None # set by def set_net_param()
        # 初始默认全不连接
        self.tp_adjacency_matrix = np.zeros((self.MINER_NUM, self.MINER_NUM))
        self.network_graph = nx.Graph(self.tp_adjacency_matrix)
        # 从csv文件中读取网络拓扑
        # self.generate_topology_from_csv(readtype='coo')
        self.network_tape = [self.BlockPacketTpNet(Block(), 0, 0, 0, self)]
        self.network_tape = []

        NET_RESULT_PATH = global_var.get_net_result_path()
        with open(NET_RESULT_PATH + '\\' + 'routing_history.json', 'a+') as f:
            f.write('[')
            json.dump({"B0": {}}, f, indent=4)
            f.write(']')


    def set_net_param(self, readtype=None, TTL=None):
        """
        设置网络参数
        param:  readtype: 读取csv文件类型, 'adj'为邻接矩阵, 'coo'为coo格式的稀疏矩阵   type:str ('adj'or'coo')
                TTL: 区块的最大生存周期, 为了防止如孤立节点的存在, 或adversary日蚀攻击,
                    导致该块一直在网络中(所有节点都收到块才判定该块传播结束)            type:int    
        """
        if readtype is not None:
            self.generate_topology_from_csv(readtype)
        if TTL is not None:
            self.TTL = TTL

    def cal_delay(self, block, source_node_id, target_node_id):
        # 传输时延=块大小除带宽 且传输时延至少1轮
        transmision_delay = ceil(
            block.blocksize_byte * 8 / self.network_graph.edges[source_node_id, target_node_id]['bandwidth'])
        # transmision_delay=transmision_delay if transmision_delay>=1 else 1
        # 时延=处理时延+传输时延
        delay = self.miners[source_node_id].processing_delay + transmision_delay
        return delay


    def cal_neighbor_delays(self, block, minerid):
        """计算minerid的邻居的时延"""
        neighbor_delays = []
        for neighborid in self.miners[minerid].neighbor_list:
            delay = self.cal_delay(block, minerid, neighborid)
            neighbor_delays.append(delay)
        return neighbor_delays


    def access_network(self, newblock, minerid, round):
        """本轮新产生的链添加到network_tape\n
        param: newblock type:block
               minerid type:int
        """
        block_packet = self.BlockPacketTpNet(newblock, minerid, round, self.TTL, self)
        self.network_tape.append(block_packet)
        self.miners[minerid].receiveBlock(newblock)  # 这一条主要是防止adversary集团发出区块的代表，自己的链上没有该区块
        print('access network ', 'miner:', minerid, newblock.name, end='', flush=True) # 加[end='']是打印进度条的时候防止换行出错哈 by CY


    def normal_forward(self, from_miner, current_miner, block_packet: BlockPacketTpNet, round):
        """
        一般传播策略。接下来传播的目标为除了from_miner和本地链中已包含该块的所有neighbor矿工。
        param:
        from_miner:该block_packet的来源     type:int  (MinerID)
        current_miner:当前的矿工            type:int  (MinerID)
        block_packet:当前处理的区块数据包    type:BlockPacketTpNet
        round:当前轮数                      type:int
        """
        # 选择接下来传播的目标
        # 除了from_miner和已包含该块的所有neighbor矿工
        bp = block_packet
        next_targets = [mi for mi in self.miners[current_miner].neighbor_list 
                        if mi != from_miner and not self.miners[mi].isInLocalChain(bp.block)]
        next_delays = []
        for nexttg in next_targets:
            next_delays.append(self.cal_delay(bp.block, current_miner, nexttg))
        bp.next_miners_and_delays.extend(
            [current_miner, nexttg, nextd] for nexttg, nextd in zip(next_targets, next_delays))

        # 记录路由
        bp.routing_histroy.update({(current_miner, nexttg): [round, 0] for nexttg in next_targets})
        bp.routing_histroy[(from_miner, current_miner)][1] = round


    def diffuse(self, round):
        """传播过程
        总的思路:
        收到一个包如果自己的本地区块链中没有,就添加到receive_tape中并转发给接下来的目标
        如果这个包存在自己的本地区块链中,就不对该包进行处理

        传播的思路类似图的遍历:
        block_packet(以下用bp)中使用列表next_miners_and_delays记录路由过程中的各链路(link),
        其元素为代表一个link的list:[scm,tgm,delay]记录从scm(source_miner)到tgm(target_miner)的delay轮数,
        每轮delay值减1,直到delay为0,表示该link传播完成,将该[scm,tgm,delay]对应的index添加到trans_complete_links,
        使用receiveBlock(bp.block)使tgm接收该块,并把tgm添加到bp.reiceved_miners中,
        此时tgm(即forward中的current_miner)变为下一个传播链路的scm,对应的scm为from_miner。

        此时来到forward():
        >>选择接下来传播的target
        current_miner选择接下来要传给的next_targets,计算cal_delay,生成链路[current_miner,nexttg,nextd],并添加到bp.next_miners_and_delays中;
        >>记录路由:
        bp.routing_histroy字典,键值对(scm,tgm):[begin_round,complete_round],
        对于新加入的link,将begin_round设为当前round,complete_round设为0;对于完成的link,将对应的complete_round设为当前ruond。

        当本轮中该bp的所有操作完成后,将trans_complete_links中对应的link从next_miners_and_delays中删除。
        最后,当bp.reiceved_miners中包含了所有的矿工或超过了TTL,表示该block传播完成,将其从network_tape中删除,并在json文件中记录其路由。
        """

        died_packets = []  # 记录已广播到所有矿工的块
        if self.network_tape:
            for i, bp in enumerate(self.network_tape):
                if len(set(bp.received_miners)) < self.MINER_NUM and bp.TTL > 0:
                    # ##判断是否都收到了####这一条代表仅适用于没有孤立节点或多个不相交子网的网络

                    trans_complete_links = []  # 记录已传播完成的链路

                    for j, [scm, tgm, delay] in enumerate(bp.next_miners_and_delays):
                        # bp.next_miners_and_delays[j][2] -= 1
                        if delay <= 0:
                            trans_complete_links.append(j)
                            # print(trans_complete_links)
                            # 处理敌对玩家
                            # if not self.miners[tgm].receiveBlock(bp.block) and self.miners[tgm].isAdversary:
                            #     #received_miners中把所有ad加上
                            #     for m,miner in enumerate(self.miners):
                            #         if miner.isAdversary:
                            #             bp.received_miners.append(m)
                            #     #邻居节点中把敌对玩家去掉
                            #     neighbor_not_ad=[n for n in self.miners[tgm].neighbor_list if not self.miners[n].isAdversary]
                            #     neighbor_not_ad_delays=[]
                            #     for n in neighbor_not_ad:
                            #         neighbor_not_ad_delays.append(self.cal_delay(bp.block,tgm,n))

                            #     #添加下一个节点
                            #     bp.next_miners_and_delays.extend([tgm,nexttg,nextd] for nexttg, nextd in zip(neighbor_not_ad,neighbor_not_ad_delays))
                            #     #记录路由
                            #     bp.routing_histroy.update({(tgm,target):[round,round+1] for target in self.miners[tgm].neighbor_list})
                            #     bp.routing_histroy.update({(tgm,target):[round,0] for target in neighbor_not_ad})
                            #     bp.routing_histroy[(scm,tgm)][1]=round

                            # 不是敌对玩家
                            # if not self.miners[tgm].receiveBlock(bp.block) and not self.miners[tgm].isAdversary:

                            if self.miners[tgm].receiveBlock(bp.block):  # if the block not in local chain, addchain,
                                bp.received_miners.append(tgm)
                                self.normal_forward(scm, tgm, bp, round)  # 传播策略
                        else:
                            # print([bp.next_miners_and_delays for bp in self.network_tape])
                            bp.next_miners_and_delays[j][2] -= 1

                    bp.next_miners_and_delays = [n for j, n in enumerate(bp.next_miners_and_delays) if
                                                 j not in trans_complete_links]
                    trans_complete_links.clear()
                    bp.TTL -= 1
                else:
                    # 对应len(set(bp.received_miners))>=self.MINER_NUM or bp.TTL<=0
                    # 所有人都收到了或该bp的生存周期结束，该block传播结束
                    # 该block加入到died_packets中
                    # 同时将其路由信息写入json文件中
                    died_packets.append(i)
                    # 将路由结果记录在json文件中
                    self.write_routing_to_json(bp)

            self.network_tape = [n for i, n in enumerate(self.network_tape) if i not in died_packets]
            died_packets.clear()

    # 带宽区块大小和时延
    # 包括固定时延和传播时延

    def read_adj_from_csv_undirected(self):
        """
        读取无向图的邻接矩阵
        """
        # 行是from 列是to
        try:
            topology_dataframe = pd.read_csv('network_topolpgy.csv', header=None, index_col=None)

            topology_ndarray = topology_dataframe.values  # 邻接矩阵，type：ndarray
            if np.isnan(topology_ndarray).any():  # 邻接矩阵中有nan，存在缺失
                raise NetAdjError('无向图邻接矩阵不规范!(存在缺失)')
            if topology_ndarray.shape[0] != topology_ndarray.shape[1]:  # 不是方阵
                raise NetAdjError('无向图邻接矩阵不规范!(row!=column)')
            if len(topology_ndarray) != self.MINER_NUM:  # 行数与环境定义的矿工数量不同
                raise NetMinerNumError('矿工数量与环境定义不符!')
            else:
                self.tp_adjacency_matrix = np.zeros((len(topology_ndarray), len(topology_ndarray)))
                for i in range(len(topology_ndarray)):
                    for j in range(i, len(topology_ndarray)):
                        if topology_ndarray[i, j] != topology_ndarray[j, i]:
                            print(i, j)  # 邻接矩阵不对称
                            raise NetAdjError(
                                '无向图邻接矩阵不规范!(row:{},column:{},{})'.format(i, j, type(topology_ndarray[i, j])))
                        if i == j:
                            if topology_ndarray[i, j] != 0:  # 邻接矩阵对角元素不为0
                                print(i, j)
                                raise NetAdjError('无向图邻接矩阵不规范!(row:{},column:{})'.format(i, j))
                        else:
                            if topology_ndarray[i, j] != 0:
                                self.tp_adjacency_matrix[i, j] = self.tp_adjacency_matrix[j, i] = 1
        except (NetMinerNumError, NetAdjError) as e:
            print(e)
            sys.exit(0)


    def generate_topology_from_csv(self, readtype='adj'):
        """
        根据csv文件的邻接矩'adj'或coo稀疏矩阵'coo'生成网络拓扑
        如果读取的是'adj',则固定每个节点间的带宽为4200000bit/round即0.5MB/round"
        如果是'coo',则带宽由用户规定
        """

        """如果读取的是邻接矩阵,则固定每个节点间的带宽为4200000bit/round即0.5MB/round"""
        if readtype == 'adj':
            # 使用pandas读取csv文件的邻接矩阵
            self.read_adj_from_csv_undirected()
            # 根据邻接矩阵生成无向图
            # bandwidth单位:bit/round
            self.network_graph = nx.Graph()
            # self.network_graph.add_nodes_from(self.miners)
            # self.network_graph.nodes
            self.network_graph.add_nodes_from([i for i in range(self.MINER_NUM)])
            for source_node in range(len(self.tp_adjacency_matrix)):  # 生成边
                for target_node in range(source_node, len(self.tp_adjacency_matrix)):
                    if self.tp_adjacency_matrix[source_node, target_node] == 1:
                        #    self.network_graph.add_edge(u,v,bandwidth=round(np.random.rand(),2))
                        self.network_graph.add_edge(source_node, target_node, bandwidth=4200000)
                        # 固定带宽为4200000bit/round,即1轮最多传输0.5MB=1048576*8*0.5=4194304 bit
        """如果读取的是coo格式稀疏矩阵"""
        if readtype == 'coo':
            # 第一行是行(from)
            # 第二行是列(to)(在无向图中无所谓from to)
            # 第三行是bandwidth
            tp_coo_dataframe = pd.read_csv('network_topolpgy_coo.csv', header=None, index_col=None)
            tp_coo_ndarray = tp_coo_dataframe.values
            row = np.array([int(i) for i in tp_coo_ndarray[0]])
            col = np.array([int(i) for i in tp_coo_ndarray[1]])
            bw_arrary = np.array([int(eval(str(i))) for i in tp_coo_ndarray[2]])
            tp_bw_coo = sp.coo_matrix((bw_arrary, (row, col)), shape=(10, 10))
            adj_values = np.array([1 for _ in range(len(bw_arrary) * 2)])
            self.tp_adjacency_matrix = sp.coo_matrix((adj_values, (np.hstack([row, col]), np.hstack([col, row]))),
                                                     shape=(10, 10)).todense()
            print('edges: \n', tp_bw_coo)
            
            self.network_graph.add_nodes_from([i for i in range(self.MINER_NUM)])
            for i in range(len(row)):
                self.network_graph.add_edge(row[i], col[i], bandwidth=bw_arrary[i])
        """read from csv finished"""

        # 邻居节点保存到各miner的neighbor_list中
        for minerid in list(self.network_graph.nodes):
            self.miners[minerid].neighbor_list = list(self.network_graph.neighbors(int(minerid)))

        # 结果展示和保存
        # print(self.network_graph.edges, self.network_graph.get_edge_data(0, 1))
        # print(self.network_graph.edges, nx.get_edge_attributes(self.network_graph, 'bandwidth'))
        print('adjacency_matrix: \n', self.tp_adjacency_matrix,'\n')
        self.draw_and_save_network()
    

    def draw_and_save_network(self):
        """
        展示和保存网络拓扑图self.network_graph
        """
        plt.ion()
        self.draw_pos = nx.spring_layout(self.network_graph, seed=50)
        # plt.figure(figsize=(12,10))
        nx.draw(self.network_graph, self.draw_pos, with_labels=True)
        node_colors = ["red" if self.miners[n].isAdversary else '#1f78b4'  for n,d in self.network_graph.nodes(data=True)]
        nx.draw_networkx_nodes(self.network_graph, pos=self.draw_pos,node_color=node_colors)
        edge_labels = {}
        for source_node, target_node in self.network_graph.edges:
            edge_labels[(source_node, target_node)] = 'BW:{}'.format(
                self.network_graph.get_edge_data(source_node, target_node)[
                    'bandwidth'])  # G[edge[0]][edge[1]] will return all attributes of edge

        nx.draw_networkx_edge_labels(self.network_graph, self.draw_pos, edge_labels=edge_labels, font_size=8,
                                     font_family='times new roman')

        RESULT_PATH = global_var.get_net_result_path()
        plt.savefig(RESULT_PATH + '\\' + 'network topology.svg')
        plt.pause(1)
        plt.close()
        plt.ioff()


    def write_routing_to_json(self, block_packet):
        """
        每当一个block传播结束,将其路由结果记录在json文件中
        json文件包含origin_miner和routing_histroy两种信息
        """
        bp = block_packet
        with open(self.NET_RESULT_PATH + '\\' + 'routing_history.json', 'a+') as f:
            f.seek(f.tell() - 1, os.SEEK_SET)
            f.truncate()
            f.write(',')
            bp.routing_histroy = {str(k): bp.routing_histroy[k] for k in bp.routing_histroy}
            json.dump({str(bp.block.name): {'origin_miner': bp.minerid, 'routing_histroy': bp.routing_histroy}}, f,
                      indent=2)
            f.write(']')


    def gen_routing_gragh_from_json(self):
        """
        读取Result->Network Routing文件夹下的routing_histroy.json,并将其转化为routing_gragh
        """
        print('Generate routing gragh for each block from json...')
        NET_RESULT_PATH = global_var.get_net_result_path()
        with open(NET_RESULT_PATH + '\\' + 'routing_history.json', 'r') as load_obj:
            # a = json.load(load_obj)
            a = json.load(load_obj)
            for v_dict in a:
                for blockname, origin_routing_dict in v_dict.items():
                    if blockname != 'B0':
                        for k, v in origin_routing_dict.items():
                            if k == 'origin_miner':
                                origin_miner = v
                            if k == 'routing_histroy':
                                rh = v
                                rh = {tuple(eval(ki)): rh[ki] for ki, _ in rh.items()}
                        self.gen_routing_gragh(blockname, rh, origin_miner)
        print('Routing gragh finished')


    def gen_routing_gragh(self, blockname, routing_histroy_single_block, origin_miner):
        """
        对单个区块生成路由图routing_gragh
        """
        # 生成有向图
        trans_process_graph = nx.DiGraph()

        trans_process_graph.add_nodes_from(self.network_graph.nodes)

        for (source_node, target_node), strounds in routing_histroy_single_block.items():
            trans_process_graph.add_edge(source_node, target_node, trans_histroy=strounds)

        # 画图和保存结果
        # plt.figure(figsize=(12, 10))
        # nx.draw(trans_process_graph,self.draw_pos,with_labels=True)
        node_colors = []
        # print([miner.isAdversary for miner in self.miners])
        # print(id(self.miners))
        for n, d in trans_process_graph.nodes(data=True):
            # print(n, type(n), self.miners[n].isAdversary)

            if self.miners[n].isAdversary and n != origin_miner:
                node_colors.append("red")
            # elif self.miners[n].isAdversary and n == origin_miner:
            #     node_colors.append("green")
            elif n == origin_miner:
                node_colors.append("green")
            else:
                node_colors.append('#1f78b4')
        node_size=200
        nx.draw_networkx_nodes(trans_process_graph, pos=self.draw_pos, node_size=node_size,node_color=node_colors)
        nx.draw_networkx_labels(trans_process_graph, pos=self.draw_pos, font_family='times new roman')
        edge_labels = {(u, v): f'{u}-{v}:{d["trans_histroy"]}' for u, v, d in trans_process_graph.edges(data=True) if
                       (v, u) not in trans_process_graph.edges()}
        edge_labels.update(dict(
            [((u, v), f'{u}-{v}:{d["trans_histroy"]}\n\n{v}-{u}:{trans_process_graph.edges[(v, u)]["trans_histroy"]}')
             for u, v, d in trans_process_graph.edges(data=True) if v > u and (v, u) in trans_process_graph.edges()]))
        
        # nx.draw_networkx_edges(trans_process_graph,self.draw_pos)
        edges_not_complete_single = [(u, v) for u, v, d in trans_process_graph.edges(data=True) if d["trans_histroy"][1] == 0 
                and (v, u) not in  trans_process_graph.edges()]
        edges_not_complete_double = [(u, v) for u, v, d in trans_process_graph.edges(data=True) if d["trans_histroy"][1] == 0 
                and (v, u) in  trans_process_graph.edges() and u>v]
        edges_complete = [ed for ed in [(u, v) for u, v, d in trans_process_graph.edges(data=True) if
                          (u, v)not in edges_not_complete_single and (u, v) not in edges_not_complete_double 
                          and (v, u) not in edges_not_complete_double]]
        # print(edges_not_complete_single,edges_not_complete_double,edges_complete)
        # print(edges_not_complete, edges_complete)
        # nx.draw_networkx_edges(trans_process_graph,self.draw_pos,edgelist=edges_not_complete,edge_color='red',alpha=0.3)
        nx.draw_networkx_edges(trans_process_graph, self.draw_pos, edgelist=edges_complete, edge_color='black', alpha=1,node_size=node_size)
        nx.draw_networkx_edges(trans_process_graph, self.draw_pos, edgelist=edges_not_complete_single, edge_color='black',
                               style='--', alpha=0.3,node_size=node_size)
        nx.draw_networkx_edges(trans_process_graph, self.draw_pos, edgelist=edges_not_complete_double, edge_color='black',
                               style='--', alpha=0.3,node_size=node_size,arrowstyle='<|-|>')                      
        nx.draw_networkx_edge_labels(trans_process_graph, self.draw_pos, edge_labels=edge_labels, font_size=5,
                                     font_family='times new roman')

        edges_list_extra = []
        for (u, v) in self.network_graph.edges:
            if ((u, v) not in trans_process_graph.edges) and ((v, u) not in trans_process_graph.edges):
                trans_process_graph.add_edge(u, v, trans_histroy=None)
                edges_list_extra.append((u, v))
        nx.draw_networkx_edges(trans_process_graph, pos=self.draw_pos, edgelist=edges_list_extra, edge_color='black',
                               alpha=0.3, style='--', arrows=False)

        NET_RESULT_PATH = global_var.get_net_result_path()
        # plt.show()
        plt.savefig(NET_RESULT_PATH + '\\' + 'transmisson process{}.svg'.format(blockname))
        plt.close()


'''    def generate_topology_random(self):
        """
        随机生成网络拓扑
        """
        # 随机生成无向图的邻接矩阵
        self.tp_adjacency_matrix = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                if i < j:
                    self.tp_adjacency_matrix[i, j] = np.random.randint(0, 2)
                if i > j:
                    self.tp_adjacency_matrix[i, j] = self.tp_adjacency_matrix[j, i]
        print(self.tp_adjacency_matrix)

        # 根据邻接矩阵生成图
        self.network_graph = nx.Graph()
        # 建立边和对应的延迟
        for i in range(len(self.tp_adjacency_matrix)):
            for j in range(len(self.tp_adjacency_matrix)):
                if self.tp_adjacency_matrix[i, j] != 0:
                    self.network_graph.add_edge(i, j, bandwidth=np.random.randint(1, 6))

        # 结果展示和保存
        print(self.network_graph.edges, nx.get_edge_attributes(self.network_graph, 'bandwidth'))
        self.draw_and_save_network()'''



if __name__ == '__main__':
    class A(object):
        def __init__(self):
            ...

        def set_net_param(self,readtype=None, TTL=None,maxround=None):
            """
            设置网络参数
            param:  readtype: 读取csv文件类型, 'adj'为邻接矩阵, 'coo'为coo格式的稀疏矩阵
                    TTL: 区块的最大生存周期, 为了防止如孤立节点的存在或
            """
            if readtype is not None:
                self.rd =readtype
            if TTL is not None:
                self.ttl= TTL
            if maxround is not None:
                self.maxr=maxround
            return(self.rd,self.ttl,self.maxr)
    a=A()
    print(a.set_net_param(TTL=9,maxround=6))

    # from miner import Miner
    # import matplotlib.pyplot as plt
    # global_var.__init__()
    # global_var.set_miner_num(10)
    # miners = []

    # miner_num=10

    # miner_i = 0
    # while miner_i < miner_num:
    #     miners.append(Miner(miner_i, 1, 1))
    #     miner_i = miner_i + 1

    # network=TopologyNetwork(miners)
    # plt.plot()
    # digraph=nx.DiGraph()
    # digraph.add_edge(1,2,t='a')
    # digraph.add_edge(2,1,t='b')
    # digraph.add_edge(3,1,t='b')
    # pos=nx.spring_layout(digraph)
    # nx.draw(digraph,pos,with_labels=True)
    # print(digraph.edges())
    # edge_labels = {}
    # edge_labels = {(u,v):f'{u}-{v}:{d["t"]}' for u,v,d in digraph.edges(data=True) if (v,u) not in digraph.edges()}
    # edge_labels.update(dict([((u, v), f'{u}-{v}:{d["t"]}\n\n{v}-{u}:{digraph.edges[(v,u)]["t"]}')
    #             for u, v, d in digraph.edges(data=True) if v > u and (v,u) in digraph.edges()]))
    # # for source_node,target_node in digraph.edges:
    # #     edge_labels[(source_node,target_node)] = 'BW:{}\n'for u,v,d in digraph.edges(data=True)) 
    # nx.draw_networkx_edge_labels(digraph, pos,edge_labels=edge_labels,font_size=8)
    # plt.show()
    # network.read_coo_undirection()
    # network.draw_and_save_network()
    # network.add_node(miners)

    # class A(object):
    #     class B(object):
    #         def __init__(self,bparam,Aobj):
    #             self.bb='b'
    #             self.bparam=bparam
    #             self.aobj=Aobj
    #             self.ba=self.aobj.aa
    #             self.bain=self.aobj.ain
    #             print(2)
    #     def __init__(self,ain) -> None:
    #         self.aa='a'
    #         self.ain=ain
    #         print(1)
    #         self.btape=[self.B(0,self)]
    #     def accessnet(self,bparam):
    #         b=self.B(bparam,self)
    #         self.btape.append(b)
    #     def printb(self):
    #         print([(b.bb,b.ba,b.bain,b.bparam) for b in self.btape])

    # a=A([1,2,3])
    # a.accessnet(11)
    # a.accessnet(22)
    # a.printb()
    # print(a.btape)

    # c=[2,1,3,1,4]
    # d=[1,2,3,4,5]
    # print('ccc',len(set(c)))
    # dic={(0,1):[1,0],(0,2):[1,3]}
    # for (u,v),value in dic.items():
    #     print(u,v,value)
    # # dic['a'][1]=2
    # print(dic)
    # e=[(cc,dd) for cc,dd in zip(c,d)]
    # for i,(cc,dd) in enumerate(e):
    #     print(i,cc,dd)
    # print([(cc,dd) for cc,dd in zip(c,d)])

    # a=1
    # b=[1]
    # if [c for c in b  if c is not  a]:
    #     print([c for c in b  if c is not  a])
    # else:
    #     print('sss')

    # list1=[1,2,3,4,0]
    # list2=[00,92,42,12]
    # for i,a, b in zip(range(len(list1)),list1,list2):
    #     list1[i]-=1
    #     b-=1
    #     print(a,b)
    # list1=list(filter((lambda x:x>0),list1))
    # print(list1)
