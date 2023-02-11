'''
    全局变量
'''
import os
import time
# MINER_NUM = 10

from task import Task

def __init__():
    '''
    初始化
    '''
    current_time = time.strftime("%Y%m%d-%H%M%S")
    RESULT_PATH=os.getcwd()+'\\'+'Results'+'\\'+current_time
    os.makedirs(RESULT_PATH) 
    NET_RESULT_PATH=RESULT_PATH+'\\'+'Network Results'
    os.makedirs(NET_RESULT_PATH)
    global _var_dict
    _var_dict = {}
    _var_dict['MINER_NUM']=0
    _var_dict['POW_TARFET']=''
    _var_dict['POW_QMAX']=0
    _var_dict['CONSENSUS_TYPE']='consensus.PoW'
    _var_dict['NETWORK_TYPE']='network.FullConnectedNetwork'
    _var_dict['BLOCK_NUMBER'] = 0
    _var_dict['RESULT_PATH'] = RESULT_PATH
    _var_dict['NET_RESULT_PATH'] = NET_RESULT_PATH
    _var_dict['Attack'] = False
    _var_dict['Blocksize'] = 2
    _var_dict['DATASET_PATH'] = './datasets/mnist.npz'
    _var_dict['MODEL_TYPE'] = 'sklearn.tree.DecisionTreeClassifier'
    _var_dict['METRIC_EVALUTOR'] = 'sklearn.metrics.accuracy_score'
    _var_dict['MINIMUM_METRIC'] = 0.8
    _var_dict['BLOCK_METRIC_REQUIREMENT'] = 0.9
    _var_dict['MINIBLOCK_NUM'] = 5
    _var_dict['BAG_SCALE'] = 0.5
    _var_dict['global_task'] = None # 需要在主程序中生成一个全局任务

def set_dataset_path(dataset_path):
    '''获取npz格式数据集的路径'''
    _var_dict['DATASET_PATH'] = dataset_path

def get_dataset_path():
    '''获取npz格式数据集的路径'''
    return _var_dict['DATASET_PATH']

def set_model_type(model_type):
    '''设置模型类型 type:str'''
    _var_dict['MODEL_TYPE'] = model_type

def get_model_type():
    '''获得模型类型'''
    return _var_dict['MODEL_TYPE']

def set_metric_evaluator(metric_evaluator):
    '''设置指标评估函数 type:str'''
    _var_dict['METRIC_EVALUTOR'] = metric_evaluator

def get_metric_evaluator():
    '''获得指标评估函数'''
    return _var_dict['METRIC_EVALUTOR']

def set_mininum_metric(mininum_metric):
    '''设置最低指标要求 type:float'''
    _var_dict['MINIMUM_METRIC'] = mininum_metric

def get_mininum_metric():
    '''获得最低指标要求'''
    return _var_dict['MINIMUM_METRIC']

def set_block_metric_requirement(block_metric):
    '''设置区块指标要求'''
    _var_dict['BLOCK_METRIC_REQUIREMENT'] = block_metric

def get_block_metric_requirement():
    '''获得区块指标要求'''
    return _var_dict['BLOCK_METRIC_REQUIREMENT']

def set_miniblock_num(miniblock_num):
    '''设置miniblock数量 type:int'''
    _var_dict['MINIBLOCK_NUM'] = miniblock_num

def get_miniblock_num():
    '''获得miniblock数量'''
    return _var_dict['MINIBLOCK_NUM']

def set_bag_scale(bag_scale):
    '''设置有放回抽样集大小 type:float'''
    _var_dict['BAG_SCALE'] = bag_scale

def get_bag_scale():
    '''获得有放回抽样集大小'''
    return _var_dict['BAG_SCALE']

def set_global_task(global_task:Task):
    '''设置全局任务 type:Task'''
    _var_dict['global_task'] = global_task

def get_global_task() -> Task:
    '''获得全局任务'''
    return _var_dict['global_task']

def set_consensus_type(consensus_type):
    '''定义共识协议类型 type:str'''
    _var_dict['CONSENSUS_TYPE'] = consensus_type

def get_consensus_type():
    '''获得共识协议类型'''
    return _var_dict['CONSENSUS_TYPE']

def set_miner_num(miner_num):
    '''定义矿工数量 type:int'''
    _var_dict['MINER_NUM'] = miner_num

def get_miner_num():
    '''获得矿工数量'''
    return _var_dict['MINER_NUM']

def set_PoW_target(PoW_target):
    '''定义pow目标 type:str'''
    _var_dict['POW_TARFET'] = PoW_target

def get_PoW_target():
    '''获得pow目标'''
    return _var_dict['POW_TARFET']

def set_qmax(PoW_qmax):
    '''定义pow,每round最多hash计算次数 type:int'''
    _var_dict['POW_QMAX'] = PoW_qmax
def get_qmax():
    '''获得pow,每round最多hash计算次数'''
    return _var_dict['POW_QMAX']

def get_block_number():
    '''获得产生区块的独立编号'''
    _var_dict['BLOCK_NUMBER'] = _var_dict['BLOCK_NUMBER'] + 1
    return _var_dict['BLOCK_NUMBER']

def get_result_path():
    return _var_dict['RESULT_PATH']

def get_net_result_path():
    return _var_dict['NET_RESULT_PATH']

def set_network_type(network_type):
    '''定义网络类型 type:str'''
    _var_dict['NETWORK_TYPE'] = network_type

def get_network_type():
    '''获得网络类型'''
    return _var_dict['NETWORK_TYPE']

def activate_attacktion():
    _var_dict['Attack'] = True

def set_blocksize(blocksize):
    _var_dict['Blocksize'] = blocksize

def get_blocksize():
    return _var_dict['Blocksize']

def set_show_fig(show_fig):
    _var_dict['Show_Fig'] = show_fig

def get_show_fig():
    return _var_dict['Show_Fig']