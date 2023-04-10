'''实现Task类'''
from enum import Enum
class Task:
    '''Task是描述机器学习任务的数据结构'''
    class DatasetType(Enum):
        '''描述任务中涉及的任务类型'''
        TRAINING_SET = 0
        TEST_SET = 1
        VALIDATION_SET = 2
    def __init__(self, training_set, test_set, validation_set, metric_evaluator,
                 block_metric_requirement, model_constructor,
                 bag_scale=0.5, fee=1):
        '''
        训练集training_set：一个元组（样本矩阵，目标值），通过引用传递，内存中只保留一份
        测试集test_set：一个元组（样本矩阵，目标值），通过引用传递，内存中只保留一份
        验证集validation_set：一个元组（样本矩阵，目标值）
        性能指标评估器metric_evaluator：一个可以根据参数y_true与y_pred计算得到性能指标的函数
        区块测试集性能指标的最低要求block_metric_requirement
        弱模型的构造函数model_constructor：一个模型的构造函数，构造出的模型对象需要有fit与predict方法
        有放回抽样集大小 bag_scale：训练弱模型之前需要进行有放回抽样，抽得的样本集称为bag，bag_scale指定每个bag中样本数量与客户提供训练集中样本数量的比值
        费用fee
        客户ID client_id
        '''
        self.training_set = training_set
        self.test_set = test_set
        self.validation_set = validation_set
        self.metric_evaluator = metric_evaluator
        self.block_metric_requirement = block_metric_requirement
        self.model_constructor = model_constructor
        self.bag_scale = bag_scale
        self.fee = fee
        self.client_id = None


    def set_client_id(self, client_id:int):
        '''设置发布任务的客户ID'''
        self.client_id = client_id
