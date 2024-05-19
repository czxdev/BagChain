import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self,*, input_channels: int = 3, num_classes = 10, image_shape: tuple = (32, 32), **kwargs):
        super(SimpleCNN, self).__init__()
        width, height = image_shape
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16 * (width//4 - 3) * (height//4 - 3), 120) # 
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

if __name__ == '__main__':
    import torch
    net = SimpleCNN(input_channels=1, num_classes=62, image_shape=(28,28))
    print(net(torch.randn(2,1,28,28)).shape)
    net = SimpleCNN(input_channels=3, num_classes=10)
    print(net(torch.randn(2,3,32,32)).shape)