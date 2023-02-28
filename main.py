from Environment import Environment
import time
import logging
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
blocksize = 16

global_var.__init__()
global_var.set_consensus_type("consensus.PoW")
# global_var.set_consensus_type("consensus.PoW")
global_var.set_miner_num(n)
global_var.set_network_type("network.TopologyNetwork")
global_var.set_qmax(q)
global_var.set_blocksize(blocksize)
global_var.set_show_fig(False)
global_var.save_configuration()

# 配置日志文件
logging.basicConfig(filename=global_var.get_result_path() / 'events.log',
                    level=global_var.get_log_level(), filemode='w')

network_param = {'readtype': 'coo', 'TTL': 500}               # Topology网络参数
#                                                               # =>readtype: 读取csv文件类型, 'adj'为邻接矩阵, 'coo'为coo格式的稀疏矩阵
#                                                               # =>TTL: 区块的最大生存周期   
adversary_ids = (5, 1, 7)     # use a tuple
Z = Environment(t, q, target, network_param, *adversary_ids)


@get_time
def run():
    Z.exec(100000)

    Z.view()

run()
