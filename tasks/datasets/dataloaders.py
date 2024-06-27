import os
import urllib3
import pickle
import torch
import numpy as np
import json
import random
from glob import glob
from pathlib import Path

def shuffle(x:np.ndarray, y:np.ndarray):
    idx = np.random.permutation(x.shape[0])
    return x[idx], y[idx]

def mnist_loader(dataset_path: Path):
    # Load MNIST dataset whose test set has been shuffled
    mnist_path = dataset_path / "mnist/mnist_testset_shuffled.npz"
    if not os.path.exists(mnist_path):
        mnist_path = dataset_path / "mnist/mnist.npz"
        if not os.path.exists(mnist_path):
            http = urllib3.PoolManager()
            print('Downloading mnist dataset to', str(mnist_path))
            response = http.request('GET',
                                    'https://s3.amazonaws.com/img-datasets/mnist.npz')
            mnist_file = open(dataset_path,'wb+')
            mnist_file.write(response.data)
            mnist_file.flush()
            response.close()
            http.clear()
            mnist_file.close()
    mnist = np.load(mnist_path)
    test_set_size = len(mnist['y_test'])
    x_train, x_test = mnist['x_train'], mnist['x_test']
    x_train = np.reshape(x_train, (x_train.shape[0],
                                   x_train.shape[1]*x_train.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0],
                                 x_test.shape[1]*x_test.shape[2]))
    training_set = (x_train, mnist['y_train'])
    test_set = (x_test[0:test_set_size//2], mnist['y_test'][0:test_set_size//2])
    validation_set = (x_test[test_set_size//2:], mnist['y_test'][test_set_size//2:])
    return (training_set, test_set, validation_set)


def load_data(data_path):
    with open(data_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    
    return batch[b'data'], batch[b'labels']

'''
from PIL import Image
import torchvision.transforms as transforms
def preprocessing(x:np.ndarray, num_channels = 3, image_shape = (32,32)) -> torch.Tensor:
        image_transform = transforms.Compose([transforms.ToTensor(), 
                                          transforms.Normalize([0.5]*num_channels,[0.5]*num_channels)])
        images = []
        width, height = image_shape
        for i in range(x.shape[0]):
            image_tensor = image_transform(Image.fromarray(x.reshape(-1,num_channels,width, height).transpose((0,2,3,1))[i]))
            images.append(image_tensor)

        x_tensor = torch.stack(images)
        return x_tensor
'''
        
from ..models import NNClassifier
PREDICT_BATCH_SIZE = NNClassifier.PREDICT_BATCH_SIZE

def datasetloader_preload(dataset, nn_params: dict = None):
    x_test, y_test = dataset
    x_test_tensor = NNClassifier.preprocessing(x_test, **(nn_params or {}))
    test_set = torch.utils.data.TensorDataset(x_test_tensor)
    test_loader = torch.utils.data.DataLoader(test_set,
                                            batch_size=PREDICT_BATCH_SIZE,
                                            shuffle=False)
    return test_loader, y_test

def cifar_loader(dataset_path: Path):
    # Load CIFAR-10 dataset whose test set has been shuffled
    cifar_path = dataset_path / 'cifar-10-batches-py'
    x_train = []
    y_train = []
    for i in range(1,6):
        x_batch, y_batch = load_data(cifar_path / ('data_batch_'+str(i)))
        x_train.append(x_batch)
        y_train.extend(y_batch)

    y_train = np.array(y_train, dtype='int64')
    x_train = np.concatenate(x_train,axis=0)
    training_set = (x_train, y_train)

    test_batch_name = 'test_batch_shuffled' if os.path.exists(cifar_path / 'test_batch_shuffled') else 'test_batch'
    x_test, y_test = load_data(cifar_path / test_batch_name)
    y_test = np.array(y_test)
    x_validation, y_validation = x_test[5000:10000], y_test[5000:10000]
    x_test, y_test = x_test[0:5000], y_test[0:5000]

    test_set = (x_test, y_test)
    validation_set = (x_validation, y_validation)

    return (training_set, test_set, validation_set)

def femnist_loader(dataset_path: Path, node_num, global_ratio):
    '''node_num: the number of nodes in the FL network
       global_ratio: the ratio of global data in the training set assigned to each node'''
    # train/test data should be contained in one single json file as the train/test split mode is by user
    dataset_path = dataset_path / 'femnist' / 'sf0-1_dataset'
    with open(next(dataset_path.glob('./train/*.json')), 'r') as f:
        train_data = json.load(f)
    with open(next(dataset_path.glob('./test/*.json')), 'r') as f:
        test_data = json.load(f)

    # merge all the data in test data into a test set
    x_test = []
    y_test = []
    for user in test_data['users']:
        x_test.extend(test_data['user_data'][user]['x'])
        y_test.extend(test_data['user_data'][user]['y'])
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_test, y_test = shuffle(x_test, y_test)
    test_sample_size = x_test.shape[0]//2
    test_set = (x_test[:test_sample_size], y_test[:test_sample_size])
    validation_set = (x_test[test_sample_size:], y_test[test_sample_size:])

    # split the users into node_num groups and merge the data in each group into a train set
    x_train = []
    y_train = []
    random.shuffle(train_data['users'])
    # generate a split of users
    user_split = [train_data['users'][i::node_num] for i in range(node_num)]
    for users in user_split:
        x_train_user = []
        y_train_user = []

        for user in users:
            x_train_user.extend(train_data['user_data'][user]['x'])
            y_train_user.extend(train_data['user_data'][user]['y'])
        x_train_user, y_train_user = shuffle(np.array(x_train_user), np.array(y_train_user))
        x_train.append(x_train_user)
        y_train.append(y_train_user)

    # calculate the size of global data
    training_data_size = sum(train_data['num_samples'])
    global_data_size = int(global_ratio * training_data_size / (node_num - global_ratio*node_num + global_ratio))
    #print('number of global samples at each node:', global_data_size)

    # split global_data_size/node_num samples from each node to form the global data
    # then combine the global data to the training set of each node in a complemental way
    x_global = []
    y_global = []
    for node in range(node_num):
        x_global.append(x_train[node][:global_data_size//node_num, :])
        y_global.append(y_train[node][:global_data_size//node_num])
        x_train[node] = x_train[node][global_data_size//node_num:, :]
        y_train[node] = y_train[node][global_data_size//node_num:] # exclude public data

    x_global, y_global = np.concatenate(x_global, axis=0), np.concatenate(y_global, axis=0)
    
    # pick some data samples to compensate the global data and make sure each category is contained in the global dataset
    num_classes = len(np.unique(y_test))
    labels = np.unique(y_global)
    node_ptr = 0
    if len(labels) < num_classes:
        for label in range(num_classes):
            if label not in labels:
                while len(idx := np.where(y_train[node_ptr] == label)[0]) == 0:
                    node_ptr = (node_ptr + 1) % node_num
                # add one sample of the missing class
                x_global = np.concatenate((x_global, x_train[node_ptr][idx[:1]]))
                y_global = np.concatenate((y_global, y_train[node_ptr][idx[:1]]))
                x_train[node_ptr] = np.delete(x_train[node_ptr], idx[:1], axis=0)
                y_train[node_ptr] = np.delete(y_train[node_ptr], idx[:1], axis=0)
                node_ptr = (node_ptr + 1) % node_num
    
    for node in range(node_num):
        x_train[node] = np.concatenate([x_train[node], x_global], axis=0)
        y_train[node] = np.concatenate([y_train[node], y_global], axis=0)
        x_train[node], y_train[node] = shuffle(x_train[node], y_train[node])
    
    train_set = [(x, y) for x, y in zip(x_train, y_train)]

    return train_set, test_set, validation_set, (x_global, y_global)

def svhn_loader(dataset_path: Path):
    from scipy.io import loadmat
    svhn_path = dataset_path / 'svhn_cropped'
    train_data = loadmat(str(svhn_path / 'train_32x32.mat'))
    test_data = loadmat(str(svhn_path / 'test_32x32_shuffled.mat'))

    x_train = np.transpose(train_data['X'], (3, 2, 0, 1))
    y_train = train_data['y'].flatten()
    np.place(y_train, y_train == 10, 0)

    x_test = np.transpose(test_data['X'], (3, 2, 0, 1))
    y_test = test_data['y'].flatten()
    np.place(y_test, y_test == 10, 0)

    training_set = (x_train, y_train)
    # Split the test set into test set and validation set
    test_sample_size = x_test.shape[0]//2
    test_set = (x_test[:test_sample_size], y_test[:test_sample_size])
    validation_set = (x_test[test_sample_size:], y_test[test_sample_size:])

    return (training_set, test_set, validation_set)