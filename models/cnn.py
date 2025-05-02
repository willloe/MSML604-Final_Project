import torch.nn as nn
from torchvision.models import resnet18

def ResNet18_CIFAR10():
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # CIFAR-10 is 32x32, skip the initial maxpool
    model.fc = nn.Linear(512, 10)
    return model
