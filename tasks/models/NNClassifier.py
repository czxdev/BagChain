import torch
from torch import nn
from torch import optim
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from PIL import Image
import gc

from .simple_cnn import SimpleCNN
from .googlenet import GoogLeNet
from .resnet import ResNet18

def reclaim_vram():
    '''Clear VRAM and RAM for a new model'''
    gc.collect()
    torch.cuda.empty_cache()

class NumpyDataset(VisionDataset):
    def __init__(self, x:np.ndarray, y:np.ndarray = None, transform = None):
        super().__init__(None, None, transform=transform)
        self.data = x
        self.target = y
        if len(self.data.shape) == 4:
            self.img_mode = 'RGB'
        elif len(self.data.shape) == 2:
            self.img_mode = 'L'
        else:
            raise ValueError("Invalid data shape")
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index:int):
        if self.target is not None:
            return self.transform(Image.fromarray(self.data[index], mode=self.img_mode)), self.target[index]
        else:
            return self.transform(Image.fromarray(self.data[index], mode=self.img_mode))

class NNClassifier():
    '''NNClassifier for Cifar10'''
    TRAINING_BATCH_SIZE = 64
    PREDICT_BATCH_SIZE = 128
    EPOCH = 10
    NN_MODEL = SimpleCNN
    MIX_PRECISION = True
    
    @staticmethod
    def preprocessing(x:np.ndarray, y:np.ndarray = None, *, input_channels = 3, image_shape = (32,32),
                      mean = None, std = None, train = False, **kwargs) -> NumpyDataset:
        '''The default values here is for Cifar10 dataset, which has 3 channels and 32x32 images'''
        transform_sequence = [transforms.ToTensor(), 
                              transforms.Normalize(mean or [0.5]*input_channels, std or [0.5]*input_channels)]
        if train:
            transform_sequence = [transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip()] + transform_sequence
        IMAGE_TRANSFORM = transforms.Compose(transform_sequence)
        width, height = image_shape
        if input_channels == 1:
            x_reshape = x.reshape(-1, width, height)
        elif input_channels == 3:
            x_reshape = x.reshape(-1, input_channels, width, height).transpose((0,2,3,1))
        else:
            raise ValueError("The input_channels is not supported")
        
        return NumpyDataset(x_reshape, y, IMAGE_TRANSFORM)
    
    def __init__(self, nn_params: dict = None) -> None:
        try:
            import torch_directml
            self.device = torch_directml.device()
        except ModuleNotFoundError:
            self.device = torch.device('cuda:0')
        self.nn_params = nn_params or {}
        num_classes = self.nn_params.get('num_classes', None)
        self.classes_ = None if num_classes is None else np.arange(num_classes)
        self.net = NNClassifier.NN_MODEL(**nn_params).to(self.device)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def predict(self, x: torch.utils.data.DataLoader):
        if isinstance(x, torch.utils.data.DataLoader):
            test_loader = x
        else:
            dataset = self.preprocessing(x, None, **self.nn_params)
            test_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=NNClassifier.PREDICT_BATCH_SIZE,
                                                    shuffle=False)

        self.net.eval()
        #self.net.to(self.device)
        predict_list = []
        with torch.no_grad():
            for data in test_loader:
                # withdraw a batch and predict
                outputs = self.net(data.to(self.device))
                _, predict = outputs.max(1)
                del outputs
                predict_list.append(predict)

            result = torch.cat(predict_list).cpu().numpy()
        return self.classes_.take(result)

    def predict_proba(self, x: torch.utils.data.DataLoader):
        if isinstance(x, torch.utils.data.DataLoader):
            test_loader = x
        else:
            dataset = self.preprocessing(x, None, **self.nn_params)
            test_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=NNClassifier.PREDICT_BATCH_SIZE,
                                                    shuffle=False)
        
        self.net.eval()
        #self.net.to(self.device)
        predict_prob_list = []
        with torch.no_grad():
            for data in test_loader:
                # withdraw a batch and predict
                outputs = self.net(data.to(self.device))
                predict_prob_list.append(torch.nn.functional.softmax(outputs,dim=1))
                del outputs

            return torch.cat(predict_prob_list).cpu().numpy()

    def fit(self, x:np.ndarray, y:np.ndarray):
        if self.classes_ is None:
            self.classes_, y_encoded = np.unique(y, return_inverse=True)
        else:
            y_encoded = np.searchsorted(self.classes_, y)
        training_dataset = self.preprocessing(x, y_encoded, train=True, **self.nn_params)
        training_loader = torch.utils.data.DataLoader(training_dataset, 
                                                      batch_size=NNClassifier.TRAINING_BATCH_SIZE,
                                                      shuffle=False, num_workers=4)
        self.train(training_loader,NNClassifier.EPOCH)
        return self

    def train(self, trainloader:torch.utils.data.DataLoader, epochs):
        if self.optimizer is None:
            raise Warning("Model cannot be trained as the optimizer has been deleted")
        
        self.net.train()
        if NNClassifier.MIX_PRECISION:
            scaler = GradScaler()
        for i in range(epochs):
            for image, label in trainloader:
                self.optimizer.zero_grad()
                image, label = image.to(self.device), label.to(self.device)
                if NNClassifier.MIX_PRECISION:
                    with autocast():
                        outputs = self.net(image)
                        loss = self.loss_function(outputs, label)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    outputs = self.net(image)
                    loss = self.loss_function(outputs, label)
                    loss.backward()
                    self.optimizer.step()

        # self.net.cpu() # make room for training of other models 
        del self.optimizer
        self.optimizer = None
