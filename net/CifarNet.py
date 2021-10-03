import torch
from torch import nn
from net.ResNet import ResNet50
import torch.nn.functional as F
from torchvision.models import resnext50_32x4d


def split_weights(net):
    decay = []
    no_decay = []
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)
            if m.bias is not None:
                no_decay.append(m.bias)
        else: 
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)
        
    assert len(list(net.parameters())) == len(decay) + len(no_decay)
    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]

class CifarNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super(CifarNet, self).__init__()
        self.backbone = ResNet50()    
        self.fc = nn.Linear(in_features=2048, out_features=num_classes)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone(x)

        x = torch.flatten(x, 1)

        return self.fc(x)