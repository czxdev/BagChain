import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from PIL import Image
import gc

from .googlenet import GoogLeNet
from .resnet import ResNet18

def reclaim_vram():
    '''Clear VRAM and RAM for a new model'''
    gc.collect()
    torch.cuda.empty_cache()

class NNClassifier():
    '''NNClassifier for Cifar10'''
    TRAINING_BATCH_SIZE = 64
    PREDICT_BATCH_SIZE = 128
    EPOCH = 10
    NN_MODEL = ResNet18
    MIX_PRECISION = True
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
        self.classes_ = None

    def predict(self, x: torch.utils.data.DataLoader):
        if isinstance(x, torch.utils.data.DataLoader):
            test_loader = x
        else:
            x_tensor, _ = self.preprocessing(x, np.array([]))
            dataset = torch.utils.data.TensorDataset(x_tensor)
            test_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=NNClassifier.PREDICT_BATCH_SIZE,
                                                    shuffle=False)

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

            result = torch.cat(predict_list).cpu().numpy()
        return self.classes_.take(result)

    def predict_proba(self, x: torch.utils.data.DataLoader):
        if isinstance(x, torch.utils.data.DataLoader):
            test_loader = x
        else:
            x_tensor, _ = self.preprocessing(x, np.array([]))
            dataset = torch.utils.data.TensorDataset(x_tensor)
            test_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=NNClassifier.PREDICT_BATCH_SIZE,
                                                    shuffle=False)
        
        self.net.eval()
        #self.net.to(self.device)
        predict_prob_list = []
        with torch.no_grad():
            for data in test_loader:
                # withdraw a batch and predict
                outputs = self.net(data[0].to(self.device))
                predict_prob_list.append(torch.nn.functional.softmax(outputs,dim=1))
                del outputs

            return torch.cat(predict_prob_list).cpu().numpy()

    def fit(self, x:np.ndarray, y:np.ndarray):
        if self.classes_ is None:
            self.classes_, y_encoded = np.unique(y, return_inverse=True)
        x_tensor_train, y_tensor_train = self.preprocessing(x, y_encoded)
        training_dataset = torch.utils.data.TensorDataset(x_tensor_train, y_tensor_train)
        training_loader = torch.utils.data.DataLoader(training_dataset, 
                                                      batch_size=NNClassifier.TRAINING_BATCH_SIZE,
                                                      shuffle=False)
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
