'''A utility function to get the mean and standard deviation of each dataset.'''
from tasks import *
from pathlib import Path
import numpy as np

def get_mean_std(x):
    '''获取数据集的均值和标准差'''
    # axis = 1 is the channel axis
    mean = []
    std = []
    
    if len(x.shape) == 2:
        if x.size == 7840000:
            x = x.reshape(-1, 28, 28)
            return [np.mean(x)], [np.std(x)]
        elif x.size == 30720000:
            x = x.reshape(-1, 3, 32, 32)
            num_channel = x.shape[1]
            for i in range(num_channel):
                mean.append(np.mean(x[:,i,:,:]))
                std.append(np.std(x[:,i,:,:]))
            return mean, std
        else:
            return [np.mean(x)], [np.std(x)]
    elif len(x.shape) == 3:
        x = x.reshape(-1, 28, 28)
        return [np.mean(x)], [np.std(x)]
    else:
        num_channel = x.shape[1]
        for i in range(num_channel):
            mean.append(np.mean(x[:,i,:,:]))
            std.append(np.std(x[:,i,:,:]))
        return mean, std


if __name__ == "__main__":
    loader = femnist_loader # mnist_loader, femnist_loader, cifar_loader, svhn_loader
    if loader == femnist_loader:
        _, test_set, validation_set = loader(Path.cwd() / 'tasks' / 'datasets', 10, 0.1)
    else:
        _, test_set, validation_set = loader(Path.cwd() / 'tasks' / 'datasets')

    x_test, y_test = test_set
    x_validation, y_validation = validation_set

    x_test = np.concatenate([x_test, x_validation], axis=0)

    print(get_mean_std(x_test))
    
