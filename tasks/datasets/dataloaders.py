import os
import urllib3
import pickle
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms


def mnist_loader(dataset_path: Path):
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

def preprocessing(x:np.ndarray) -> torch.Tensor:
        image_transform = transforms.Compose([transforms.ToTensor(), 
                                          transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
        images = []
        for i in range(x.shape[0]):
            image_tensor = image_transform(Image.fromarray(x.reshape(-1,3,32,32).transpose((0,2,3,1))[i]))
            images.append(image_tensor)

        x_tensor = torch.stack(images)
        return x_tensor

PREDICT_BATCH_SIZE = 2048

def cifar_loader(dataset_path: Path):
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

    x_test_tensor = preprocessing(x_test)
    test_set = torch.utils.data.TensorDataset(x_test_tensor)
    test_loader = torch.utils.data.DataLoader(test_set,
                                            batch_size=PREDICT_BATCH_SIZE,
                                            shuffle=False)
    test_set = test_loader, y_test

    x_validation_tensor = preprocessing(x_validation)
    validation_set = torch.utils.data.TensorDataset(x_validation_tensor,)
    validation_loader = torch.utils.data.DataLoader(validation_set,
                                            batch_size=PREDICT_BATCH_SIZE,
                                            shuffle=False)
    validation_set = validation_loader, y_validation

    return (training_set, test_set, validation_set)