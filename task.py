'''实现Task类'''
from enum import Enum
from tasks import cifar_loader, mnist_loader, femnist_loader, svhn_loader, femnist_loader_iid
import global_var
from functions import for_name

class Task:
    '''Task是描述机器学习任务的数据结构'''
    class DatasetType(Enum):
        '''描述任务中涉及的任务类型'''
        TRAINING_SET = 0
        TEST_SET = 1
        VALIDATION_SET = 2
    def __init__(self, training_set, test_set, validation_set, global_dataset,
                 metric_evaluator, block_metric_requirement, model_constructor,
                 model_params, bag_scale=0.5, fee=1):
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
        self.global_dataset = global_dataset
        self.metric_evaluator = metric_evaluator
        self.block_metric_requirement = block_metric_requirement
        self.model_constructor = model_constructor
        self.model_params = model_params
        self.bag_scale = bag_scale
        self.fee = fee
        self.client_id = None


    def set_client_id(self, client_id:int):
        '''设置发布任务的客户ID'''
        self.client_id = client_id

    def get_dataset(self, miner_id, dataset_type):
        '''获取数据集'''
        if dataset_type == Task.DatasetType.TRAINING_SET:
            if type(self.training_set) == tuple:
                return self.training_set
            elif type(self.training_set) == list and len(self.training_set) > miner_id:
                return self.training_set[miner_id]
            else:
                raise ValueError("The length of training set sequence is erroneous")
        elif dataset_type == Task.DatasetType.TEST_SET:
            return self.test_set
        elif dataset_type == Task.DatasetType.VALIDATION_SET:
            return self.validation_set
        else:
            raise ValueError("Invalid dataset type")

def model_importer(model_name):
    '''根据模型名导入模型'''
    if model_name == "DTC":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier
    else:
        if model_name == "CNN":
            from tasks.models import SimpleCNN as NN
        elif model_name == "GoogLeNet":
            from tasks.models import GoogLeNet as NN
        elif model_name == "ResNet18":
            from tasks.models import ResNet18 as NN
        elif model_name == "MobileNetV2":
            from tasks.models import MobileNetV2 as NN
        elif model_name == "DenseNet":
            from tasks.models import DenseNet121_12 as NN
        else:
            raise ValueError("MODEL match none of the following: DTC, CNN, GoogLeNet, ResNet18")
        from tasks.models import NNClassifier
        NNClassifier.NN_MODEL = NN
    return NNClassifier

def global_task_init(selection:str, noniid_conf: dict = None, epoch: int = 10):
    '''
        为Task获取数据集、性能评估函数与模型
        selection=A: MNIST + Decision Tree Classifier As base model
        selection=B: CIFAR10 + GoogLeNet As base model
        selection=C-[DATASET]-[MODEL]: federated learning tasks with imbalance datasets
            DATASET: the training set shall be a sequence of local datasets
                Distribution based label imbalance: MNIST, CIFAR10, SVHN(?)
                Realistic label imbalance: FEMNIST
            MODEL: DTC, CNN, GoogLeNet (DTC is only used in )
        noniid_conf: a dictionary containing the configuration of non-iid data distribution
    '''
    dataset_path = global_var.get_dataset_path()

    noniid_conf = noniid_conf or {}
    global_var.set_bag_scale(1) # TODO 默认bag_scale为1把DTC的结果重跑一遍
    nn_params = {}
    if selection.startswith("A-") or selection == "A":# DTC iid
        selection = selection.split("-")
        if len(selection) > 2:
            raise ValueError("Selection of task is invalid")
        if len(selection) == 1 or len(selection) == 2 and selection[1] == "MNIST":
            training_set, test_set, validation_set = mnist_loader(dataset_path)
            dataset = "MNIST"
            block_metric = 0.8
        elif len(selection) == 2 and selection[1] == "FEMNIST":
            training_set, test_set, validation_set, global_dataset \
                     = femnist_loader(dataset_path, global_var.get_miner_num(), 1.0)
            dataset = "FEMNIST"
            block_metric = 1/62
        else:
            raise ValueError("Don't use DTC with RGB images, use CNN instead, or the performance shall be terrible")

        metric_evaluator = for_name(global_var.get_metric_evaluator())
        model_constructor = for_name("sklearn.tree.DecisionTreeClassifier")
        model = "DTC"
        
    elif selection.startswith("B-") or selection == "B": # NNClassifier iid
        selection = selection.split("-")
        if len(selection) == 2 or len(selection) > 3:
            raise ValueError("Selection of task is invalid")
        if len(selection) == 1:
            training_set, test_set, validation_set = cifar_loader(dataset_path)
            dataset = "CIFAR10"
            model = "ResNet18"
            block_metric = 0.1
        elif len(selection) == 3:
            dataset = selection[1]
            model = selection[2]
            if model == "DTC":
                raise ValueError("To use DTC, selection = A")
            if dataset == "MNIST":
                training_set, test_set, validation_set = mnist_loader(dataset_path)
                block_metric = 0.1
                nn_params = {'input_channels': 1, 'image_shape': (28, 28), 'num_classes': 10,
                            'mean': [0.13251460584233693], 'std': [0.310480247930535]}

            elif dataset == "CIFAR10":
                training_set, test_set, validation_set = cifar_loader(dataset_path)
                block_metric = 0.1
                nn_params = {'input_channels': 3, 'image_shape': (32, 32), 'num_classes': 10,
                            'mean': [0.4942142800245098, 0.4851313890165441, 0.4504090927542892],
                            'std': [0.24665251509497996, 0.24289226346005366, 0.2615923780220232]}
            elif dataset == "FEMNIST":
                training_set, test_set, validation_set = femnist_loader_iid(dataset_path)
                block_metric = 1/62
                nn_params = {'input_channels': 1, 'image_shape': (28, 28), 'num_classes': 62,
                            'mean': [0.9638689148893337], 'std': [0.15864969199187845]}
            elif dataset == "SVHN":
                training_set, test_set, validation_set = svhn_loader(dataset_path)
                block_metric = 0.1
                nn_params = {'input_channels': 3, 'image_shape': (32, 32), 'num_classes': 10,
                            'mean': [0.4524231572700963, 0.4524958429274829, 0.4689771312228746],
                            'std': [0.21943445421025629, 0.22656966836656006, 0.228506126737217]}
            else:
                raise ValueError("DATASET match none of the following: MNIST, CIFAR10, FEMNIST, SVHN")
        else:
            raise ValueError("Selection of task is invalid")
        model_constructor = model_importer(model)
        metric_evaluator = for_name(global_var.get_metric_evaluator())

    elif selection.startswith("C-"):
        from tasks import partition_label_distribution, partition_label_quantity, partition_by_index
        selection = selection.split("-")
        if len(selection) != 3:
            raise ValueError("Selection of task is invalid")
        dataset = selection[1]
        model = selection[2]
        miner_num = global_var.get_miner_num()
        capable_miner_num = noniid_conf.get('capable_miner_num') or miner_num
        # Load datasets
        if dataset == "MNIST":
            training_set, test_set, validation_set = mnist_loader(dataset_path)
            block_metric = 0.1
            nn_params = {'input_channels': 1, 'image_shape': (28, 28), 'num_classes': 10,
                         'mean': [0.13251460584233693], 'std': [0.310480247930535]}
                         # 'mean': [33.791224489795916], 'std': [79.17246322228642]
        elif dataset == "CIFAR10":
            training_set, test_set, validation_set = cifar_loader(dataset_path)
            block_metric = 0.1
            nn_params = {'input_channels': 3, 'image_shape': (32, 32), 'num_classes': 10,
                         'mean': [0.4942142800245098, 0.4851313890165441, 0.4504090927542892],
                         'std': [0.24665251509497996, 0.24289226346005366, 0.2615923780220232]}
                         #'mean': [126.02464140625, 123.70850419921875, 114.85431865234375],
                         #'std': [62.89639134921989, 61.93752718231368, 66.7060563956159]
        elif dataset == "FEMNIST":
            training_set, test_set, validation_set, global_dataset \
                     = femnist_loader(dataset_path, capable_miner_num, noniid_conf['global_ratio'])
            block_metric = 1/62
            nn_params = {'input_channels': 1, 'image_shape': (28, 28), 'num_classes': 62,
                         'mean': [0.9638689148893337], 'std': [0.15864969199187845]}
        elif dataset == "SVHN":
            training_set, test_set, validation_set = svhn_loader(dataset_path)
            block_metric = 0.1
            nn_params = {'input_channels': 3, 'image_shape': (32, 32), 'num_classes': 10,
                         'mean': [0.4524231572700963, 0.4524958429274829, 0.4689771312228746],
                          'std': [0.21943445421025629, 0.22656966836656006, 0.228506126737217]}
                         #'mean': [115.36790510387456, 115.38643994650815, 119.58916846183303],
                         #'std': [55.95578582361535, 57.77526543347282, 58.26906231799034]
        else:
            raise ValueError("DATASET match none of the following: MNIST, CIFAR10, FEMNIST, SVHN")
        
        # Split datasets into non-iid datasets
        if dataset != "FEMNIST": # Not needed for FEMNIST
            # Generate a global dataset
            # global_ratio: the ratio of global sample counts over the total sample count at each node
            from tasks import generate_global_dataset
            global_dataset, training_set = generate_global_dataset(training_set, noniid_conf['global_ratio'],
                                                                   miner_num, nn_params['num_classes'])
            y_train = training_set[1]
            
            if noniid_conf['type'] == "label_quantity":
                data_index = partition_label_quantity(noniid_conf['label_per_miner'], capable_miner_num, 
                                                      nn_params['num_classes'], y_train)
            elif noniid_conf['type'] == "label_distribution":
                data_index = partition_label_distribution(noniid_conf['beta'], capable_miner_num, 
                                                          nn_params['num_classes'], y_train)
            else:
                raise ValueError("Invalid non-iid data distribution type")
            training_set = partition_by_index(training_set, global_dataset, capable_miner_num,
                                              miner_num, data_index)
        elif capable_miner_num < miner_num: # when not all miners are capable,
                                            # replicate the global dataset for non-capable miners
            training_set.extend([global_dataset] * (miner_num-capable_miner_num))
            if len(training_set) != miner_num:
                raise ValueError("The length of training set sequence is erroneous")

        # Load models and metric evaluators
        metric_evaluator = for_name(global_var.get_metric_evaluator())
        model_constructor = model_importer(model)

    else:
        raise ValueError("Selection of task is invalid")

    if model != "DTC": # NNClassifier
        # Preload the test and validation dataset into DataLoader
        from tasks.datasets.dataloaders import datasetloader_preload
        test_set = datasetloader_preload(test_set, nn_params)
        validation_set = datasetloader_preload(validation_set, nn_params)
        # set epoch
        model_constructor.EPOCH = epoch
    # 构建Task对象
    def construct_model():
        if model == "DTC":
            return model_constructor()
        else:
            return model_constructor(nn_params)
    use_global = global_dataset if noniid_conf.get('base_global_experiment') else None
    task1 = Task(training_set, test_set, validation_set, use_global, metric_evaluator,
                block_metric, construct_model, nn_params, global_var.get_bag_scale())
    task1.set_client_id(0)
    global_var.set_global_task(task1)
    with open(global_var.get_result_path() / "task_params.txt", "w") as f:
        f.write("Task Params: " + "\n" + \
                "Model: " + model + "\n" + "Dataset: " + dataset + "\n" + \
                "Non-iid Configuration: " + str(noniid_conf) + "\n" + \
                "Bag Scale: " + str(global_var.get_bag_scale()) + "\n" + \
                "Block Metric Requirement: " + str(block_metric) + "\n")
