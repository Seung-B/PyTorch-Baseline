import torch
import torch.nn as nn
import torchvision.models as models


class SimpleMLP(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28 * input_channels, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28 * x.size(1))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_resnet(model_name, input_channels, num_classes, pretrain):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrain)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrain)

    if input_channels != 3:
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def get_densenet(input_channels, num_classes, pretrained):
    model = models.densenet121(pretrained=pretrained)

    if input_channels != 3:
        conv0 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            conv0.weight[:, :] = torch.unsqueeze(model.features.conv0.weight.mean(1), 1)
        model.features.conv0 = conv0

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    return model
