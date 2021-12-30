import torch
from torch import nn
from torchvision import transforms

from model import ConvNet

from custom_dataset_loader import StrawberryDataset
from torch.utils.data import DataLoader

# Setting the device for using the CUDA if is possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setting hyperparameters
input_size = None
number_of_classes = 7
learning_rate = 0.001
batch_size = 32
num_epoch = 10

# tensor([0.3778, 0.4980, 0.1993]) -> mean
# tensor([0.1722, 0.1499, 0.1384]) -> std

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.3778, 0.4980, 0.1993], [0.1722, 0.1499, 0.1384])])

# Loading dateset (loading training, testing and validating sets)
train_set = StrawberryDataset(csv_file='dataset/data_train.csv', image_dir='dataset/train',
                              transform=transforms.ToTensor())
test_set = StrawberryDataset(csv_file='dataset/data_test.csv', image_dir='dataset/test',
                             transform=transform)
val_set = StrawberryDataset(csv_file='dataset/data_val.csv', image_dir='dataset/val',
                            transform=transform)

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epoch}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
