import random

from .network_abc import Network

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
                    miner.receive_block(newblock)

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
                                    self.miners[j].receive_block(block_packet['block'])
                                    # print(block_packet['block'].name,block_packet['recieve_prob'],block_packet['received_miner'],file=f)
                                    # 如果接收到了区块，则通过receiveBlock函数传递给该miner
                                    # if not miners[j].receiveBlock(block_packet['block']):#判断最近是否收到过该区块

                                # 判断如果是isAdversary:
                                # 如果一个ad收到，其他ad也立即收到
                                if self.miners[j].isAdversary:
                                    block_packet['received_miner'].append(self.miners[j].Miner_ID)
                                    block_packet['received_round'].append(round)
                                    self.miners[j].receive_block(block_packet['block'])
                                    for miner in self.miners:
                                        if miner.isAdversary and miner.Miner_ID != j:
                                            block_packet['received_miner'].append(miner.Miner_ID)
                                            block_packet['received_round'].append(round)
                                            miner.receive_block(block_packet['block'])

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