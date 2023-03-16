"""network module 实现网络功能"""
import json
import os
import random
import sys
import logging
from abc import ABCMeta, abstractmethod
from math import ceil

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

import errors
import global_var
from chain import Block, BlockHead


logger = logging.getLogger(__name__)

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

        with open(self.NET_RESULT_PATH / 'network_log.txt', 'a') as f:
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



class BoundedDelayNetwork(Network):
    """矿工以概率接收到区块，在特定轮数前必定所有矿工都收到区块"""

    def __init__(self, miners: list):
        super().__init__()
        self.miners = miners
        self.network_tape = []
        self.rcvprob_start = 0.25
        self.rcvprob_inc = 0.25
        # 结果打印准备
        with open(self.NET_RESULT_PATH / 'network_log.txt', 'a') as f:
            print('Network Type: BoundedDelayNetwork', file=f)


    def set_net_param(self, rcvprob_start, rcvprob_inc):
        """设置网络参数\n
        param:  rcvprob_start:每个包进入网络时的接收概率,默认0.25
                rcvprob_inc:之后每轮增加的接收概率,默认0.25"""
        self.rcvprob_start = rcvprob_start
        self.rcvprob_inc = rcvprob_inc
        with open(self.NET_RESULT_PATH / 'network_log.txt', 'a') as f:
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

        with open(self.NET_RESULT_PATH / 'network_log.txt', 'a') as f:

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
        # 结果保存路径
        NET_RESULT_PATH = global_var.get_net_result_path()
        with open(NET_RESULT_PATH / 'routing_history.json', 'a+') as f:
            f.write('[')
            json.dump({"B0": {}}, f, indent=4)
            f.write(']')


    def set_net_param(self, gen_net_approach = None, TTL = None, save_routing_graph = None, edge_prob = None, show_label = None):
        """设置网络参数
        param:  gen_approach: 生成网络的方式, 'adj'为邻接矩阵, 'coo'为coo格式的稀疏矩阵   type:str ('adj'or'coo')
                TTL: 区块的最大生存周期, 为了防止如孤立节点的存在, 或adversary日蚀攻击,
                    导致该块一直在网络中(所有节点都收到块才判定该块传播结束)            type:int    
        """
        if show_label is not None:
            self.show_label = show_label
        if gen_net_approach is not None:
            if  gen_net_approach == 'rand' and edge_prob is not None:
                self.gen_net_approach = gen_net_approach
                self.edge_prob = edge_prob
                self.generate_network(gen_net_approach,edge_prob)
            else:
                self.gen_net_approach = gen_net_approach
                self.edge_prob = None
                self.generate_network(gen_net_approach)
        if TTL is not None:
            self.TTL = TTL
        if save_routing_graph is not None:
            self.save_routing_graph = save_routing_graph
            

    def access_network(self, newblock, minerid, round):
        """本轮新产生的链添加到network_tape\n
        param: newblock type:block
               minerid type:int
        """
        block_packet = self.BlockPacketTpNet(newblock, minerid, round, self.TTL, self)
        self.network_tape.append(block_packet)
        self.miners[minerid].receiveBlock(newblock)  # 这一条主要是防止adversary集团发出区块的代表，自己的链上没有该区块
        # print('access network ', 'miner:', minerid, newblock.name, end='', flush=True) # 加[end='']是打印进度条的时候防止换行出错哈 by CY
        logger.info("access network miner:%d %s at round %d", minerid, newblock.name, round)


    def diffuse(self, round):
        """传播过程
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
        died_packets = []  # 记录传播结束的块
        if self.network_tape:
            for i, bp in enumerate(self.network_tape):
                if len(set(bp.received_miners)) < self.MINER_NUM and bp.TTL > 0:
                    # 判断是否都收到了,这一条代表diffuse仅适用于没有孤立节点或子网的网络
                    trans_complete_links = []  # 记录已传播完成的链路
                    for j, [scm, tgm, delay] in enumerate(bp.next_miners_and_delays):
                        if delay <= 0:
                            self.receive_forward(scm, tgm, bp, round)
                            trans_complete_links.append(j)
                        else:
                            bp.next_miners_and_delays[j][2] -= 1
                    if trans_complete_links:#在next_miners_and_delays中删掉已传播完成的link
                        bp.next_miners_and_delays = [n for j, n in enumerate(bp.next_miners_and_delays) 
                                                     if j not in trans_complete_links]
                    trans_complete_links.clear()
                    bp.TTL -= 1
                else:# 所有人都收到了或该bp的生存周期结束，该block传播结束
                    died_packets.append(i)# 该block加入died_packets
                    self.write_routing_to_json(bp)# 将路由结果记录在json文件中
            if died_packets:# 在network_tape中清理掉传播结束的块
                self.network_tape = [n for i, n in enumerate(self.network_tape) if i not in died_packets]
                died_packets.clear()


    def receive_forward(self, from_miner, current_miner, block_packet: BlockPacketTpNet, round):
        """接收并转发区块
        收到一个包,本地链上没有,就添加到receive_tape中并转发给接下来的目标
        否则不对该包进行处理
        """
        if self.miners[current_miner].receiveBlock(block_packet.block) is True:# if the block not in local chain, receive.
            block_packet.received_miners.append(current_miner) # 记录已接收的矿工
            self.normal_forward(from_miner, current_miner, block_packet, round)  # 执行转发策略


    def normal_forward(self, from_miner, current_miner, block_packet: BlockPacketTpNet, round):
        """一般转发策略。接下来转发的目标为除了from_miner和本地链中已包含该块的所有neighbor矿工。
        param:
        from_miner:该block_packet的来源     type:int  (MinerID)
        current_miner:当前的矿工            type:int  (MinerID)
        block_packet:当前处理的区块数据包    type:BlockPacketTpNet
        round:当前轮数                      type:int
        """
        # 选择接下来赚翻的目标--除了from_miner和已包含该块的所有neighbor矿工
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


    def cal_delay(self, block, sourceid, targetid):
        """计算sourceid和targetid之间的时延"""
        # 传输时延=块大小除带宽 且传输时延至少1轮
        transmision_delay = ceil(
            block.blocksize_byte * 8 / self.network_graph.edges[sourceid, targetid]['bandwidth'])
        # 时延=处理时延+传输时延
        delay = self.miners[sourceid].processing_delay + transmision_delay
        return delay


    def cal_neighbor_delays(self, block, minerid):
        """计算minerid的邻居的时延"""
        neighbor_delays = []
        for neighborid in self.miners[minerid].neighbor_list:
            delay = self.cal_delay(block, minerid, neighborid)
            neighbor_delays.append(delay)
        return neighbor_delays
    

    def generate_network(self, gen_net_approach, edge_prop=None):
        """
        根据csv文件的邻接矩'adj'或coo稀疏矩阵'coo'生成网络拓扑    
        """
        # read from csv finished
        try:
            if gen_net_approach == 'adj':
                self.gen_network_adj()
            elif gen_net_approach == 'coo':
                self.gen_network_coo()
            elif gen_net_approach == 'rand' and edge_prop is not None:
                self.gen_network_rand(edge_prop)
            else:raise errors.NetGenError('网络生成方式错误！')
            #检查是否有孤立节点或不连通部分
            if not nx.number_of_isolates(self.network_graph):
                if nx.number_connected_components(self.network_graph) == 1:
                    # 邻居节点保存到各miner的neighbor_list中
                    for minerid in list(self.network_graph.nodes):
                        self.miners[minerid].neighbor_list = list(self.network_graph.neighbors(int(minerid)))
                    # 结果展示和保存
                    #print('adjacency_matrix: \n', self.tp_adjacency_matrix,'\n')
                    self.draw_and_save_network()
                    self.save_network_attribute()
                else:
                    raise errors.NetUnconnetedError('网络存在不连通部分!')
            else:raise errors.NetIsoError('网络存在孤立节点! {}'.format(list(nx.isolates(self.network_graph))))
        except (errors.NetMinerNumError, errors.NetAdjError, errors.NetIsoError, 
                errors.NetUnconnetedError, errors.NetGenError) as e:
            print(e); sys.exit(0)
    
    def save_network_attribute(self):
        network_attributes={
            'miner_num':self.MINER_NUM,
            'Generate Approach':self.gen_net_approach,
            'Generate Edge Probability':self.edge_prob,
            'Diameter':nx.diameter(self.network_graph),
            'Average Shortest Path Length':nx.average_shortest_path_length(self.network_graph),
            'Degree Histogram': nx.degree_histogram(self.network_graph),
            'Average Cluster Coefficient':nx.average_clustering(self.network_graph),
            'Degree Assortativity':nx.degree_assortativity_coefficient(self.network_graph),
        }
        NET_RESULT_PATH = global_var.get_net_result_path()
        with open(NET_RESULT_PATH / 'Network Attributes.txt', 'a+') as f:
            f.write('Network Attributes')
            for k,v in network_attributes.items():
                f.write(str(k)+': '+str(v)+'\n')
        print(network_attributes)

    def gen_network_rand(self, edge_prop):
        """采用Erdős-Rényi算法生成随机图"""
        self.network_graph = nx.gnp_random_graph(self.MINER_NUM, edge_prop)
        
        bandwidths = {(u,v):(4200000*10 if self.miners[u].isAdversary and  
                             self.miners[v].isAdversary else 4200000) for u,v in self.network_graph.edges}
        nx.set_edge_attributes(self.network_graph, bandwidths, "bandwidth")
        self.tp_adjacency_matrix = nx.adjacency_matrix(self.network_graph).todense()


    def gen_network_adj(self):
        """
        如果读取邻接矩阵,则固定节点间的带宽为4200000bit/round即0.5MB/round
        bandwidth单位:bit/round
        """
        # 读取csv文件的邻接矩阵
        self.read_adj_from_csv_undirected()
        # 根据邻接矩阵生成无向图
        self.network_graph = nx.Graph()
        self.network_graph.add_nodes_from([i for i in range(self.MINER_NUM)])
        for source_node in range(len(self.tp_adjacency_matrix)):  # 生成边
            for target_node in range(source_node, len(self.tp_adjacency_matrix)):
                if self.tp_adjacency_matrix[source_node, target_node] == 1:
                    self.network_graph.add_edge(source_node, target_node, bandwidth=4200000)
                    # 固定带宽为4200000bit/round,即1轮最多传输0.5MB=1048576*8*0.5=4194304 bit


    def gen_network_coo(self):
        """如果读取'coo'稀疏矩阵,则带宽由用户规定"""
        # 第一行是行(from)
        # 第二行是列(to)(在无向图中无所谓from to)
        # 第三行是bandwidth:bit/round
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


    def read_adj_from_csv_undirected(self):
        """读取无向图的邻接矩阵adj"""
        # 行是from 列是to
        topology_dataframe = pd.read_csv('network_topolpgy.csv', header=None, index_col=None)
        topology_ndarray = topology_dataframe.values  # 邻接矩阵，type：ndarray
        if np.isnan(topology_ndarray).any():  # 邻接矩阵中有nan，存在缺失
            raise errors.NetAdjError('无向图邻接矩阵不规范!(存在缺失)')
        if topology_ndarray.shape[0] != topology_ndarray.shape[1]:  # 不是方阵
            raise errors.NetAdjError('无向图邻接矩阵不规范!(row!=column)')
        if len(topology_ndarray) != self.MINER_NUM:  # 行数与环境定义的矿工数量不同
            raise errors.NetMinerNumError('矿工数量与环境定义不符!')
        else:
            self.tp_adjacency_matrix = np.zeros((len(topology_ndarray), len(topology_ndarray)))
            for i in range(len(topology_ndarray)):
                for j in range(i, len(topology_ndarray)):
                    if topology_ndarray[i, j] != topology_ndarray[j, i]: # 不为对称阵
                        raise errors.NetAdjError(
                            '无向图邻接矩阵不规范!(row:{},column:{},{})'.format(i, j, type(topology_ndarray[i, j])))
                    if i == j:
                        if topology_ndarray[i, j] != 0:  # 邻接矩阵对角元素不为0
                            raise errors.NetAdjError('无向图邻接矩阵不规范!(row:{},column:{})'.format(i, j))
                    else:
                        if topology_ndarray[i, j] != 0:
                            self.tp_adjacency_matrix[i, j] = self.tp_adjacency_matrix[j, i] = 1


    def draw_and_save_network(self):
        """
        展示和保存网络拓扑图self.network_graph
        """
        plt.ion()
        self.draw_pos = nx.spring_layout(self.network_graph, seed=50)
        # plt.figure(figsize=(12,10))
        node_size=200*3/self.MINER_NUM**0.5
        #nx.draw(self.network_graph, self.draw_pos, with_labels=True,node_size=node_size,font_size=30/(self.MINER_NUM)^0.5,width=3/self.MINER_NUM)
        node_colors = ["red" if self.miners[n].isAdversary else '#1f78b4'  for n,d in self.network_graph.nodes(data=True)]
        nx.draw_networkx_nodes(self.network_graph, pos=self.draw_pos,node_color=node_colors,node_size=node_size)
        nx.draw_networkx_labels(self.network_graph, pos=self.draw_pos,font_size=30/self.MINER_NUM**0.5,font_family='times new roman')
        edge_labels = {}
        for source_node, target_node in self.network_graph.edges:
            edge_labels[(source_node, target_node)] = 'BW:{}'.format(
                self.network_graph.get_edge_data(source_node, target_node)[
                    'bandwidth'])  # G[edge[0]][edge[1]] will return all attributes of edge
        nx.draw_networkx_edges(self.network_graph, pos=self.draw_pos,width=3/self.MINER_NUM**0.5,node_size=node_size)
        if self.show_label:
            nx.draw_networkx_edge_labels(self.network_graph, self.draw_pos, edge_labels=edge_labels, font_size=12/self.MINER_NUM**0.5,
                                        font_family='times new roman')

        RESULT_PATH = global_var.get_net_result_path()
        plt.savefig(RESULT_PATH / 'network topology.svg')
        plt.pause(1)
        plt.close()
        plt.ioff()


    def write_routing_to_json(self, block_packet):
        """
        每当一个block传播结束,将其路由结果记录在json文件中
        json文件包含origin_miner和routing_histroy两种信息
        """
        bp = block_packet
        with open(self.NET_RESULT_PATH / 'routing_history.json', 'a+') as f:
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
        if self.save_routing_graph == False:
            print('Fail to generate routing gragh for each block from json.')
        elif self.save_routing_graph == True:  
            print('Generate routing gragh for each block from json...')
            NET_RESULT_PATH = global_var.get_net_result_path()
            with open(NET_RESULT_PATH / 'routing_history.json', 'r') as load_obj:
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
        # 处理节点颜色
        node_colors = []
        for n, d in trans_process_graph.nodes(data=True):
            if self.miners[n].isAdversary and n != origin_miner:
                node_colors.append("red")
            elif n == origin_miner:
                node_colors.append("green")
            else:
                node_colors.append('#1f78b4')
        node_size=100*3/self.MINER_NUM**0.5
        nx.draw_networkx_nodes(trans_process_graph, pos=self.draw_pos, node_size=node_size,node_color=node_colors)
        nx.draw_networkx_labels(trans_process_graph, pos=self.draw_pos, font_size=30/self.MINER_NUM**0.5,font_family='times new roman')
        # 对边进行分类，分为单向未传播完成的、双向未传播完成的、已传播完成的
        edges_not_complete_single = [(u, v) for u, v, d in trans_process_graph.edges(data=True) if d["trans_histroy"][1] == 0 
                and (v, u) not in  trans_process_graph.edges()]
        edges_not_complete_double = [(u, v) for u, v, d in trans_process_graph.edges(data=True) if d["trans_histroy"][1] == 0 
                and (v, u) in  trans_process_graph.edges() and u>v]
        edges_complete = [ed for ed in [(u, v) for u, v, d in trans_process_graph.edges(data=True) if
                          (u, v)not in edges_not_complete_single and (u, v) not in edges_not_complete_double 
                          and (v, u) not in edges_not_complete_double]]
        # 画边
        width=3/self.MINER_NUM**0.5
        nx.draw_networkx_edges(trans_process_graph, self.draw_pos, edgelist=edges_complete, edge_color='black', 
                               width=width,alpha=1,node_size=node_size,arrowsize=30/self.MINER_NUM**0.5)
        nx.draw_networkx_edges(trans_process_graph, self.draw_pos, edgelist=edges_not_complete_single, edge_color='black',
                               width=width,style='--', alpha=0.3,node_size=node_size,arrowsize=30/self.MINER_NUM**0.5)
        nx.draw_networkx_edges(trans_process_graph, self.draw_pos, edgelist=edges_not_complete_double, edge_color='black',
                               width=width,style='--', alpha=0.3,node_size=node_size,arrowstyle='<|-|>',arrowsize=30/self.MINER_NUM**0.5)                      
        # 没有传播到的边用虚线画
        edges_list_extra = []
        for (u, v) in self.network_graph.edges:
            if ((u, v) not in trans_process_graph.edges) and ((v, u) not in trans_process_graph.edges):
                trans_process_graph.add_edge(u, v, trans_histroy=None)
                edges_list_extra.append((u, v))
        nx.draw_networkx_edges(trans_process_graph, pos=self.draw_pos, edgelist=edges_list_extra, edge_color='black',
                                width=3/self.MINER_NUM**0.5,alpha=0.3, style='--', arrows=False)
        # 处理边上的label，对单向的双向和分别处理
        if self.show_label:
            edge_labels = {(u, v): f'{u}-{v}:{d["trans_histroy"]}' for u, v, d in trans_process_graph.edges(data=True) if
                        (v, u) not in trans_process_graph.edges()}
            edge_labels.update(dict(
                [((u, v), f'{u}-{v}:{d["trans_histroy"]}\n\n{v}-{u}:{trans_process_graph.edges[(v, u)]["trans_histroy"]}')
                for u, v, d in trans_process_graph.edges(data=True) if v > u and (v, u) in trans_process_graph.edges()]))
            nx.draw_networkx_edge_labels(trans_process_graph, self.draw_pos, edge_labels=edge_labels, font_size=5*2/self.MINER_NUM**0.5,
                                font_family='times new roman')
            
        #保存svg图片
        NET_RESULT_PATH = global_var.get_net_result_path()
        plt.savefig(NET_RESULT_PATH / ('routing_graph{}.svg'.format(blockname)))
        plt.close()

    def calculate_stats(self):
        stats = {
            "average_network_delay" : 0
        }
        delay_list = []
        RECEIVED_RATIO = 1 # 网络中收到区块的矿工数量相对所有矿工数量之比如果超过这一常量，计算网络延迟
        miner_num = len(self.miners)
        NET_RESULT_PATH = global_var.get_net_result_path()
        with open(NET_RESULT_PATH / 'routing_history.json', 'r') as load_obj:
            a = json.load(load_obj)
            for v_dict in a[1:]:
                for _, origin_routing_dict in v_dict.items():
                    routing_history = origin_routing_dict["routing_histroy"]
                    miner_received = 0
                    access_network_time = 0
                    for transmision,time_span in routing_history.items():
                        arrival_time = time_span[1]
                        if tuple(eval(transmision))[0] == origin_routing_dict["origin_miner"]:
                            access_network_time = time_span[0]
                        if arrival_time:
                            miner_received += 1
                            if miner_received + 1>= miner_num * RECEIVED_RATIO:
                                delay_list.append(arrival_time - access_network_time)
        stats['average_network_delay'] = sum(delay_list)/len(delay_list)
        return stats


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


'''
class BoundedDelayNetwork(Network):
    """全连接BoundedDelay广播"""

    def __init__(self, miners: list):
        super().__init__()
        self.miners = miners
        self.miner_delaylist = []  # 各miner的网络delay，元素int:经过多少轮才能被其他miner接收到,
        self.network_tape = []
        self.genMinerDelay(5)  # 最大网络时延为5

        with open(self.NET_RESULT_PATH / 'network_log.txt', 'a') as f:
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
if __name__ == '__main__':
    for i in np.arange(0.1,1.1,0.1):
        print(i)


