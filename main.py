'''主程序'''
import time
import numpy as np
import os
import urllib3
import logging
# from sklearn.datasets import fetch_openml

import global_var
import configparser
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

def global_var_init(n, q, blocksize, miniblock_size, result_path = None):
    global_var.__init__(result_path)
    global_var.set_consensus_type("pob.PoB")
    # global_var.set_consensus_type("consensus.PoW")
    global_var.set_miner_num(n)
    global_var.set_network_type("network.TopologyNetwork")
    global_var.set_ave_q(q)
    global_var.set_blocksize(blocksize)
    global_var.set_miniblock_size(miniblock_size)
    global_var.set_show_fig(False)

def global_task_init():
    # 为Task获取数据集
    dataset_path = global_var.get_dataset_path()
    if not os.path.exists(dataset_path):
        http = urllib3.PoolManager()
        print('Downloading mnist dataset to', str(dataset_path))
        response = http.request('GET',
                                'https://s3.amazonaws.com/img-datasets/mnist.npz')
        mnist_file = open(dataset_path,'wb+')
        mnist_file.write(response.data)
        mnist_file.flush()
        response.close()
        http.clear()
        mnist_file.close()
    mnist = np.load(dataset_path)
    test_set_size = len(mnist['y_test'])
    x_train, x_test = mnist['x_train'], mnist['x_test']
    x_train = np.reshape(x_train, (x_train.shape[0],
                                   x_train.shape[1]*x_train.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0],
                                 x_test.shape[1]*x_test.shape[2]))
    training_set = (x_train, mnist['y_train'])
    test_set = (x_test[0:test_set_size//2], mnist['y_test'][0:test_set_size//2])
    validation_set = (x_test[test_set_size//2:], mnist['y_test'][test_set_size//2:])

    # 获取性能评估函数与模型
    metric_evaluator = for_name(global_var.get_metric_evaluator())
    model = for_name(global_var.get_model_type())

    # 构建Task对象
    task1 = Task(training_set, test_set,validation_set, metric_evaluator,
                global_var.get_block_metric_requirement(), model,
                global_var.get_bag_scale())
    task1.set_client_id(0)
    global_var.set_global_task(task1)


@get_time
def run(Z, total_round, max_height) -> dict:
    Z.exec(total_round, max_height)
    return Z.view()

def main(
    total_round = 100,
    n = 10,  # number of miners
    t = 0,   # maximum number of adversary
    q = 5,
    target = '000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
    blocksize = 16,
    miniblock_size = 2,
    result_path = None,
    max_height = 1000000,
    test_set_interval = 100,
    validation_set_interval = 10,
    network_generator = "coo",
    matrix = None):

    global_var_init(n, q, blocksize, miniblock_size, result_path)
    global_var.set_PoW_target(target)
    global_var.set_block_metric_requirement(0.86)
    global_var.set_test_set_interval(test_set_interval)
    global_var.set_validation_set_interval(validation_set_interval)
    global_var.set_ensemble_block_num(20)
    global_var.set_log_level(logging.INFO)
    
    shuffled_dataset_path = global_var.get_dataset_path().parent/"mnist_testset_shuffled.npz"
    if shuffled_dataset_path.exists():
        # 如果有测试集打乱过的数据集则使用
        global_var.set_dataset_path(shuffled_dataset_path)
    global_var.save_configuration()
    global_task_init()

    # 配置日志文件
    logging.basicConfig(filename=global_var.get_result_path() / 'events.log',
                        level=global_var.get_log_level(), filemode='w')
    
    network_param = {'gen_net_approach': network_generator, 'TTL': 500, 'matrix': matrix}
 
    adversary_ids = ()     # no attacks
    return run(Environment(t, q, 'equal', target, network_param, *adversary_ids), total_round, max_height)

if __name__ == "__main__":
    main(60000, 10, blocksize=2,max_height=20)