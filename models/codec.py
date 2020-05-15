import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3t(in_planes, out_planes, stride=1, output_padding=0):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, output_padding=output_padding, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, output_padding=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes, stride=1)
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.relu(out)
        return out

class BasicBlockT(nn.Module):
    def __init__(self, inplanes, planes, stride=1, outpadding=0):
        super(BasicBlockT, self).__init__()
        self.conv1 = conv3x3t(inplanes, inplanes, stride=1)
        self.conv2 = conv3x3t(inplanes, planes, stride, outpadding)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

class CodecNet(nn.Module):
    def __init__(self, layers, class_num):
        super(CodecNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)

        #transpose conv
        self.layer_t1 = self._make_layer(BasicBlockT, 256, layers[4], stride=2, outpadding=1)
        self.layer_t2 = self._make_layer(BasicBlockT, 128, layers[5], stride=2, outpadding=1)
        self.layer_t3 = self._make_layer(BasicBlockT, 64, layers[6], stride=2, outpadding=1)
        self.maxunpool = nn.MaxUnpool2d(2, stride=2)
        self.conv_t4 = nn.ConvTranspose2d(64, class_num, kernel_size=5, stride=2, padding=2, output_padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, outpadding=0):
        layers = []
        for i in range(blocks):
            layers.append(block(self.inplanes, planes, stride, outpadding))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x, indices = self.maxpool(x)

        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)

        x4 = self.layer_t1(x3)
        x4 += x2
        x5 = self.layer_t2(x4)
        x5 += x1
        x6 = self.layer_t3(x5)
        x6 += x
        x = self.maxunpool(x6, indices)
        x = self.conv_t4(x)
        x = F.log_softmax(x, dim=1)
        return x

def CodecNet13(class_num):
    model = CodecNet([1, 1, 1, 1, 1, 1, 1, 1], class_num)
    return model
