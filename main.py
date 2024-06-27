'''主程序'''
import time
import logging

import global_var
import configparser
from Environment import Environment
from task import global_task_init

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


@get_time
def run(Z, total_round, max_height) -> dict:
    Z.exec(total_round, max_height)
    return Z.view()

def main(
    total_round = 100,*,
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
    matrix = None,
    task_selection = "A",
    noniid_conf = None):

    global_var_init(n, q, blocksize, miniblock_size, result_path)
    global_var.set_PoW_target(target)
    global_var.set_test_set_interval(test_set_interval)
    global_var.set_validation_set_interval(validation_set_interval)
    global_var.set_log_level(logging.INFO)
    global_var.save_configuration()
    global_task_init(task_selection, noniid_conf=noniid_conf)

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
    MINER_NUM = 10
    import numpy as np
    matrix = np.ones((MINER_NUM,MINER_NUM)) - np.eye(MINER_NUM)
    print(main(60000, n=MINER_NUM, blocksize=6,max_height=2,network_generator='matrix',
         matrix=matrix, task_selection='C-MNIST-DTC',
         noniid_conf={'type':'label_distribution', 'global_ratio':0.02, 'label_per_miner':3,
                      'beta':0.5, 'capable_miner_num': 1, 'base_global_experiment': True}))
    
    # Task selection: A-[DATASET], B-[DATASET]-[MODEL], C-[DATASET]-[MODEL]
    # Possible selections: A, B, C-MNIST-DTC, C-MNIST-CNN,
    #                            C-CIFAR10-CNN, C-CIFAR10-GoogLeNet, C-CIFAR10-ResNet18,
    #                            C-FEMNIST-DTC, C-FEMNIST-CNN, C-FEMNIST-GoogLeNet, C-FEMNIST-ResNet18,
    #                            C-SVHN-CNN, C-SVHN-GoogLeNet, C-SVHN-ResNet18
    # Note: DTC is not suitable for RGB images
    # noniid_conf = {'type':'label_quantity', 'global_ratio': 0.1, 'beta': 0.5,
    #                'label_per_miner': 3, 'capable_miner_num': None}
    #                'capable_miner_num' means the number of miners that own both computing power and private dataset
    #                                   If it is None, it equals to miner_num by default
    #                'base_global_experiment' means whether to output the test set metric of a model trained on the global dataset
                    