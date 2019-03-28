'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
from .modules.quantize import quantize, quantize_grad, QConv2d, QLinear, RangeBN

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = QConv2d(3, 6, 5, num_bits_weight = 8)
        self.conv2 = QConv2d(6, 16, 5, num_bits_weight = 8)
        self.fc1   = QLinear(16*5*5, 120, num_bits_weight = 8)
        self.fc2   = QLinear(120, 84, num_bits_weight = 8)
        self.fc3   = QLinear(84, 10, num_bits_weight = 8)

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
