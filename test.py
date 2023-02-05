# from Environment_VDF import Environment as Environment_V
from Environment import Environment
import time
import global_var


def get_time(f):
    def inner(*arg, **kwarg):
        s_time = time.time()
        res = f(*arg, **kwarg)
        e_time = time.time()
        print('耗时：{}秒'.format(e_time - s_time))
        return res
    return inner


n = 10  # number of miners
t = 3   # maximum number of adversary
q = 5

target = '000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF'
#           ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff43
#          'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF43'
# target = hex(50000)[2:]
# target = targetG(0.5,10,99263413,q) 
blocksize = 16

global_var.__init__()
global_var.set_consensus_type("consensus.PoW")
# global_var.set_consensus_type("Optimization.PoOptimization")
global_var.set_miner_num(n)
# global_var.set_network_type("network.BoundedDelayNetwork")
global_var.set_network_type("network.TopologyNetwork")
global_var.set_qmax(q)
global_var.set_blocksize(blocksize)
global_var.set_show_fig(False)

# network_param={"rcvprob_start":0.1, 'rcvprob_inc':0.1}         # BoundedDelay网络参数
#                                                               # =>rcvprob_start:每个包进入网络时的接收概率,默认0.25
#                                                               # =>rcvprob_inc:之后每轮增加的接收概率,默认0.25"""
network_param = {'readtype': 'coo', 'TTL': 500}               # Topology网络参数
#                                                               # =>readtype: 读取csv文件类型, 'adj'为邻接矩阵, 'coo'为coo格式的稀疏矩阵
#                                                               # =>TTL: 区块的最大生存周期   
adversary_ids = (5, 1, 7)     # use a tuple
Z = Environment(t, q, target, network_param, *adversary_ids)


@get_time
def run():
    # Z.select_adversary(*adversary_ids)
    Z.exec(1000)

    Z.view()


run()
# Z.miners[0].Blockchain.Popblock()
# print(id((Z.miners[1].Blockchain)))
# print(id((Z.miners[2].Blockchain)))
# print(id((Z.miners[3].Blockchain)))
# Z.miners[1].Blockchain.ShowStructure1()
# Z.miners[2].Blockchain=copy.deepcopy(Z.miners[1].Blockchain)    
# Z.miners[2].Blockchain.ShowStructure1()
# print('======================')
# Z.global_chain.ShowStructure1()
# for block in Z.miners[1].Blockchain:
#     block.printblock()
# print('======================')
# for block in Z.miners[2].Blockchain:
#     block.printblock()
# print('======================')
# Z.miners[1].Blockchain.lastblock.printblock() 
# Z.miners[2].Blockchain.lastblock.printblock() 
# print(1)