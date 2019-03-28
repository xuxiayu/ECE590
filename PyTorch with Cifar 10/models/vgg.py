'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from .modules.quantize import quantize, quantize_grad, QConv2d, QLinear, RangeBN

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, layer_chunk, weight_bits):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], layer_chunk, weight_bits)
        if layer_chunk == 5:
            print("FC")
            print(weight_bits)
            self.classifier = QLinear(512, 10, num_bits_weight = weight_bits)
            #, num_bits=weight_bits, num_bits_grad=weight_bits,biprecision=True
        else:
            self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, layer_chunk, weight_bits):
        layers = []
        in_channels = 3
        index = 0
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                index = index + 1
            elif index == layer_chunk and layer_chunk != 5:
                print(x)
                print(weight_bits)
                #num_bits=weight_bits, num_bits_grad=weight_bits,biprecision=True
                layers += [QConv2d(in_channels, x, kernel_size=3, padding=1, num_bits_weight = weight_bits),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
                
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
        

def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
