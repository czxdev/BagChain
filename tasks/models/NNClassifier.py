import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
from PIL import Image

from .lenet import LeNet
#from .resnet import ResNet18

class NNClassifier():
    '''NNClassifier for Cifar10'''
    TRAINING_BATCH_SIZE = 64
    PREDICT_BATCH_SIZE = 128
    EPOCH = 10
    NN_MODEL = LeNet
    IMAGE_TRANSFORM = transforms.Compose([transforms.ToTensor(), 
                                          transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    def preprocessing(self, x:np.ndarray, y:np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        images = []
        for i in range(x.shape[0]):
            image_tensor = NNClassifier.IMAGE_TRANSFORM(Image.fromarray(x.reshape(-1,3,32,32).transpose((0,2,3,1))[i]))
            images.append(image_tensor)

        x_tensor = torch.stack(images)
        y_tensor = torch.tensor(y, dtype=torch.int64)
        return x_tensor, y_tensor
    
    def __init__(self) -> None:
        self.device = torch.device("cuda:0") # Use GPU
        self.net = NNClassifier.NN_MODEL().to(self.device)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def predict(self, test_loader: torch.utils.data.DataLoader):
        self.net.eval()
        #self.net.to(self.device)
        predict_list = []
        with torch.no_grad():
            for data in test_loader:
                # withdraw a batch and predict
                outputs = self.net(data[0].to(self.device))
                _, predict = outputs.max(1)
                del outputs
                predict_list.append(predict)

        #self.net.cpu()
        return torch.cat(predict_list).cpu().numpy()

    def fit(self, x:np.ndarray, y:np.ndarray):
        x_tensor_train, y_tensor_train = self.preprocessing(x, y)
        training_dataset = torch.utils.data.TensorDataset(x_tensor_train, y_tensor_train)
        training_loader = torch.utils.data.DataLoader(training_dataset, 
                                                      batch_size=NNClassifier.TRAINING_BATCH_SIZE,
                                                      shuffle=False)
        self.train(training_loader,NNClassifier.EPOCH)
        return self

    def train(self, trainloader:torch.utils.data.DataLoader, epochs):
        if self.optimizer is None:
            raise Warning("Model cannot be trained as the optimizer has been deleted")
        train_loss = 0
        correct = 0
        total = 0
        self.net.train()
        for i in range(epochs):
            for image, label in trainloader:
                self.optimizer.zero_grad()
                image, label = image.to(self.device), label.to(self.device)
                outputs = self.net(image)
                loss = self.loss_function(outputs, label)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += label.size(0)
                correct += predicted.eq(label).sum().item()
        # self.net.cpu() # make room for training of other models 
        del self.optimizer
        self.optimizer = None