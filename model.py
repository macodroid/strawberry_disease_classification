import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # convolution layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn5 = nn.BatchNorm2d(1024)

        # pooling
        self.pool = nn.MaxPool2d(2, 2)

        # dropout
        self.dropout = nn.Dropout2d(0.5)
        self.dropoutConvLayer = nn.Dropout2d(0.1)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=1024 * 1 * 1, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=7)

    def forward(self, x):
        # Here we are connecting them

        # first layer
        x = self.conv1(x)
        x = self.dropoutConvLayer(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # second layer
        x = self.conv2(x)
        x = self.dropoutConvLayer(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # third layer
        x = self.conv3(x)
        x = self.dropoutConvLayer(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        # forth layer
        x = self.conv4(x)
        x = self.dropoutConvLayer(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)

        # fifth layer
        x = self.conv5(x)
        x = self.dropoutConvLayer(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = x.view(-1, 1024 * 1 * 1)
        x = nn.functional.relu(self.fc1(self.dropout(x)))
        x = nn.functional.relu(self.fc2(self.dropout(x)))
        x = self.fc3(x)
        return x
