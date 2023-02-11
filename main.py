'''主程序'''
import time
import numpy as np
import os
import urllib3
# from sklearn.datasets import fetch_openml

import global_var
from Environment import Environment
from functions import for_name
from task import Task

def get_time(f):
    def inner(*arg, **kwarg):
        s_time = time.time()
        res = f(*arg, **kwarg)
        e_time = time.time()
        print('耗时：{}秒'.format(e_time - s_time))
        return res
    return inner

def global_var_init(n, q, blocksize):
    global_var.__init__()
    global_var.set_consensus_type("pob.PoB")
    # global_var.set_consensus_type("consensus.PoW")
    global_var.set_miner_num(n)
    global_var.set_network_type("network.TopologyNetwork")
    global_var.set_qmax(q)
    global_var.set_blocksize(blocksize)
    global_var.set_show_fig(False)

def global_task_init():
    # 为Task获取数据集
    dataset_path = global_var.get_dataset_path()
    if not os.path.exists(dataset_path):
        http = urllib3.PoolManager()
        print('Downloading mnist dataset to', dataset_path)
        response = http.request('GET',
                                'https://s3.amazonaws.com/img-datasets/mnist.npz')
        mnist_file = open(dataset_path,'wb+')
        mnist_file.write(response.data)
        mnist_file.flush()
        response.close()
        http.clear()
        mnist_file.close()
    mnist = np.load(dataset_path)
    x_train, x_test = mnist['x_train'], mnist['x_test']
    x_train = np.reshape(x_train, (x_train.shape[0],
                                   x_train.shape[1]*x_train.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0],
                                 x_test.shape[1]*x_test.shape[2]))
    training_set = (x_train, mnist['y_train'])
    test_set = (x_test, mnist['y_test'])

    # 获取性能评估函数与模型
    metric_evaluator = for_name(global_var.get_metric_evaluator())
    model = for_name(global_var.get_model_type())

    # 构建Task对象
    task1 = Task(training_set, test_set, metric_evaluator, global_var.get_mininum_metric(),
                global_var.get_block_metric_requirement(), model,
                global_var.get_miniblock_num(), global_var.get_bag_scale())
    task1.set_client_id(0)
    global_var.set_global_task(task1)


@get_time
def run():
    Z.exec(100000)

    Z.view()

if __name__ == "__main__":
    n = 10  # number of miners
    t = 3   # maximum number of adversary
    q = 5

    target = '000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF'
    blocksize = 16

    global_var_init(n, q, blocksize)
    global_task_init()

    network_param = {'readtype': 'coo', 'TTL': 500}   # Topology网络参数
    #                                                 # =>readtype: 读取csv文件类型, 'adj'为邻接矩阵, 'coo'为coo格式的稀疏矩阵
    #                                                 # =>TTL: 区块的最大生存周期   
    adversary_ids = (5, 1, 7)     # use a tuple
    Z = Environment(t, q, target, network_param, *adversary_ids)
    run()