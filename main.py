'''主程序'''
import time
import logging

import global_var
import configparser
from Environment import Environment
from functions import for_name
from task import Task

from tasks import mnist_loader
from tasks import cifar_loader

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
    '''
        为Task获取数据集、性能评估函数与模型
        selection A: MNIST + Decision Tree Classifier As base model
        selection B: CIFAR19 + LeNet As base model
    '''
    dataset_path = global_var.get_dataset_path()
    selection = global_var.get_task_selection()
    if selection == "A":
        training_set, test_set, validation_set = mnist_loader(dataset_path)
        metric_evaluator = for_name(global_var.get_metric_evaluator())
        model = for_name("sklearn.tree.DecisionTreeClassifier")
        block_metric = 0.8
    elif selection == "B":
        training_set, test_set, validation_set = cifar_loader(dataset_path)
        from tasks.models import NNClassifier
        metric_evaluator = for_name(global_var.get_metric_evaluator())
        model = NNClassifier
        block_metric = 0.55
        global_var.set_bag_scale(1)
    else:
        raise ValueError("Selection of task is invalid")

    # 构建Task对象
    task1 = Task(training_set, test_set,validation_set, metric_evaluator,
                block_metric, model, global_var.get_bag_scale())
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
    global_var.set_test_set_interval(test_set_interval)
    global_var.set_validation_set_interval(validation_set_interval)
    global_var.set_log_level(logging.INFO)
    global_var.save_configuration()
    global_task_init()

    # 配置日志文件
    logging.basicConfig(filename=global_var.get_result_path() / 'events.log',
                        level=global_var.get_log_level(), filemode='w')
    
    # 读取配置文件
    config = configparser.ConfigParser()
    config.optionxform = lambda option: option
    config.read('system_config.ini',encoding='utf-8')

    network_param = {'gen_net_approach': network_generator, 'TTL': 500, 'matrix': matrix,
                     'show_label':True}
    if network_generator=="rand":
        network_param.update({'TTL':config.getint('TopologyNetworkSettings', 'TTL'),
                        'save_routing_graph': config.getboolean('TopologyNetworkSettings', 'save_routing_graph'),
                        'ave_degree': config.getfloat('TopologyNetworkSettings', 'ave_degree'),
                        'show_label': config.getboolean('TopologyNetworkSettings', 'show_label'),
                        'bandwidth_honest': config.getfloat('TopologyNetworkSettings', 'bandwidth_honest'),
                        'bandwidth_adv': config.getfloat('TopologyNetworkSettings', 'bandwidth_adv')
                        })
 
    adversary_ids = ()     # no attacks
    return run(Environment(t, q, 'equal', target, network_param, *adversary_ids), total_round, max_height)

if __name__ == "__main__":
    import os
    #from affinity import set_process_affinity_mask
    #set_process_affinity_mask(os.getpid(), 1<<6)
    MINER_NUM = 7
    import numpy as np
    matrix = np.ones((MINER_NUM,MINER_NUM)) - np.eye(MINER_NUM)
    main(60000, MINER_NUM, blocksize=6,max_height=15,network_generator='matrix',
         matrix=matrix)