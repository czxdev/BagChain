'''Reference: https://github.com/Xtra-Computing/NIID-Bench/blob/main/partition.py'''

import numpy as np
import random
from tasks import shuffle

def partition_label_quantity(label_per_miner, partition_num, label_number, y_train):
    contain=[]
    # 初始化 times 为全零列表
    times = [0 for _ in range(label_number)]

    # 对每个参与方，随机选择 label_per_miner 个不重复的标签
    for i in range(partition_num):
        current = random.sample(range(label_number), label_per_miner)
        for label in current:
            times[label] += 1
        contain.append(current)
    dataidx = [np.ndarray(0,dtype=np.int64) for _ in range(partition_num)]
    for i in range(label_number):
        if times[i]==0:
            continue
        idx_k = np.where(y_train==i)[0]
        np.random.shuffle(idx_k)
        split = np.array_split(idx_k,times[i])
        ids=0
        for j in range(partition_num):
            if i in contain[j]:
                dataidx[j]=np.append(dataidx[j],split[ids])
                ids+=1
    for i in range(partition_num):
        dataidx[i] = dataidx[i].tolist()
    return dataidx

def partition_label_distribution(beta, partition_num, label_number, y_train):
    min_size = 0
    min_require_size = 10

    N = y_train.shape[0]
    dataidx = [[] for _ in range(partition_num)]

    if y_train.shape[0] == 0:
        return dataidx

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(partition_num)]
        for k in range(label_number):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, partition_num))
            # logger.info("proportions1: ", proportions)
            # logger.info("sum pro1:", np.sum(proportions))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / partition_num) for p, idx_j in zip(proportions, idx_batch)])
            # logger.info("proportions2: ", proportions)
            proportions = proportions / proportions.sum()
            # logger.info("proportions3: ", proportions)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            # logger.info("proportions4: ", proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            # if K == 2 and n_parties <= 10:
            #     if np.min(proportions) < 200:
            #         min_size = 0
            #         break

    for j in range(partition_num):
        np.random.shuffle(idx_batch[j])
        dataidx[j] = idx_batch[j]
    
    return dataidx

def partition_by_index(training_set, global_dataset, capable_miner_num, total_miner_num, data_index):
    x_train, y_train = training_set
    x_global, y_global = global_dataset
    training_set_list = []
    for i in range(capable_miner_num):
        local_dataset = shuffle(np.concatenate((x_global, x_train[data_index[i]])),
                                np.concatenate((y_global, y_train[data_index[i]])))
        training_set_list.append(local_dataset)
    capable_miners = np.random.choice(range(total_miner_num),
                                      capable_miner_num, replace=False).tolist()

    miner_training_list = []
    for i in range(total_miner_num):
        if i in capable_miners:
            miner_training_list.append(training_set_list[capable_miners.index(i)])
        else:
            miner_training_list.append(global_dataset)
        
    return miner_training_list

def generate_global_dataset(training_set, global_ratio, partition_num, num_classes):
    '''sample a global dataset from the training set, make sure each class is included'''
    x_train, y_train = training_set
    training_set_size = y_train.shape[0]
    global_dataset_size = int(global_ratio * training_set_size)
    x_train, y_train = shuffle(x_train, y_train)
    x_global = x_train[:global_dataset_size]
    y_global = y_train[:global_dataset_size]
    x_train = x_train[global_dataset_size:]
    y_train = y_train[global_dataset_size:]
    labels = np.unique(y_global)
    if len(labels) < num_classes:
        for label in range(num_classes):
            if label not in labels:
                # add one sample of the missing class
                idx = np.where(y_train == label)[0]
                x_global = np.concatenate((x_global, x_train[idx[:1]]))
                y_global = np.concatenate((y_global, y_train[idx[:1]]))
                x_train = np.delete(x_train, idx[:1], axis=0)
                y_train = np.delete(y_train, idx[:1], axis=0)
    global_dataset = (x_global, y_global)
    new_training_set = (x_train, y_train)
    return global_dataset, new_training_set