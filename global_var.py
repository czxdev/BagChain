'''
    全局变量
'''
from pathlib import Path
import time
import logging
# MINER_NUM = 10

from task import Task

def __init__(result_path = None):
    '''
    初始化
    '''
    current_time = time.strftime("%Y%m%d-%H%M%S")
    RESULT_PATH=result_path or Path.cwd() / 'Results' / current_time
    RESULT_PATH.mkdir(parents=True)   
    NET_RESULT_PATH=RESULT_PATH / 'Network Results'
    NET_RESULT_PATH.mkdir()
    global _var_dict
    _var_dict = {}
    # 区块链基本参数
    _var_dict['MINER_NUM']=0
    _var_dict['POW_TARFET']=''
    _var_dict['AVE_Q']=0
    _var_dict['CONSENSUS_TYPE']='consensus.PoW'
    _var_dict['NETWORK_TYPE']='network.FullConnectedNetwork'
    _var_dict['BLOCK_NUMBER'] = 0
    _var_dict['RESULT_PATH'] = RESULT_PATH
    _var_dict['NET_RESULT_PATH'] = NET_RESULT_PATH
    _var_dict['Attack'] = False
    _var_dict['Blocksize'] = 2
    _var_dict['MINIBLOCK_SIZE'] = 2
    # 学习任务相关
    _var_dict['DATASET_PATH'] = Path.cwd() / 'datasets/mnist.npz'
    _var_dict['MODEL_TYPE'] = 'sklearn.tree.DecisionTreeClassifier'
    _var_dict['METRIC_EVALUTOR'] = 'sklearn.metrics.accuracy_score'
    _var_dict['BLOCK_METRIC_REQUIREMENT'] = 0.91
    # 系统参数
    _var_dict['TEST_SET_INTERVAL'] = 90
    _var_dict['VALIDATION_SET_INTERVAL'] = 180
    _var_dict['TASK_QUEUE_LENGTH'] = 2
    _var_dict['BAG_SCALE'] = 0.5
    _var_dict['global_task'] = None # 需要在主程序中生成一个全局任务
    _var_dict['LOG_LEVEL'] = logging.INFO
    _var_dict['Show_Fig'] = False
    _var_dict['MINIBLOCK_COUNTER'] = {'B0':-1} # 迷你块计数器，以字典形式为每个块之后的miniblock计数
    

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

def set_block_metric_requirement(block_metric):
    '''设置区块指标要求'''
    _var_dict['BLOCK_METRIC_REQUIREMENT'] = block_metric

def get_block_metric_requirement():
    '''获得区块指标要求'''
    return _var_dict['BLOCK_METRIC_REQUIREMENT']
def set_test_set_interval(interval):
    '''设置测试集发布时间间隔 type:int'''
    _var_dict['TEST_SET_INTERVAL'] = interval

def get_test_set_interval():
    '''获得测试集发布时间间隔'''
    return _var_dict['TEST_SET_INTERVAL']

def set_validation_set_interval(interval):
    '''设置验证集发布时间间隔 type:int'''
    _var_dict['VALIDATION_SET_INTERVAL'] = interval

def get_validation_set_interval():
    '''获得验证集发布时间间隔'''
    return _var_dict['VALIDATION_SET_INTERVAL']

def set_task_queue_length(task_queue_length):
    '''设置任务队列长度 type:int'''
    _var_dict['TASK_QUEUE_LENGTH'] = task_queue_length

def get_task_queue_length():
    '''获得任务队列长度'''
    return _var_dict['TASK_QUEUE_LENGTH']

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

def set_log_level(log_level):
    '''设置日志级别'''
    _var_dict['LOG_LEVEL'] = log_level

def get_log_level():
    '''获得日志级别'''
    return _var_dict['LOG_LEVEL']

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

def set_ave_q(ave_q):
    '''定义pow,每round最多hash计算次数 type:int'''
    _var_dict['AVE_Q'] = ave_q
def get_ave_q():
    '''获得pow,每round最多hash计算次数'''
    return _var_dict['AVE_Q']

def get_block_number():
    '''获得产生区块的独立编号'''
    _var_dict['BLOCK_NUMBER'] = _var_dict['BLOCK_NUMBER'] + 1
    return _var_dict['BLOCK_NUMBER']

def get_miniblock_num(pre_block_name:str):
    '''获得某一block之后的miniblock数量'''
    return _var_dict['MINIBLOCK_COUNTER'][pre_block_name]

def get_miniblock_name(pre_block_name:str) -> str:
    '''产生miniblock的名称'''
    # 检查pre_block_name的有效性
    pre_block_num = int(pre_block_name.split('B')[1])
    if pre_block_num > _var_dict['BLOCK_NUMBER']:
        raise Warning(f"Invalid block name:{pre_block_name}")
    _var_dict['MINIBLOCK_COUNTER'].setdefault(pre_block_name,-1)
    _var_dict['MINIBLOCK_COUNTER'][pre_block_name] += 1
    
    return f"{pre_block_name}-{_var_dict['MINIBLOCK_COUNTER'][pre_block_name]}"

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
    
def set_miniblock_size(size):
    '''设置miniblock大小'''
    _var_dict['MINIBLOCK_SIZE'] = size

def get_miniblock_size():
    '''得到miniblock大小'''
    return _var_dict['MINIBLOCK_SIZE']

def set_show_fig(show_fig):
    _var_dict['Show_Fig'] = show_fig

def get_show_fig():
    return _var_dict['Show_Fig']

def save_configuration():
    '''将_var_dict中的内容保存到configuration.txt中'''
    with open(_var_dict['RESULT_PATH'] / "configuration.txt",
              'w+') as config:
        for key,value in _var_dict.items():
            print(key,": ",value,file=config)
