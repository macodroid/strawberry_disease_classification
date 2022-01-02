import torch
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms

from model import ConvNet

from custom_dataset_loader import StrawberryDataset
from torch.utils.data import DataLoader

# Setting the device for using the CUDA if is possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.cuda.empty_cache()

# keeping-track-of-losses
train_losses = []
valid_losses = []
test_accuracy = 0

file_name = 'cnn_run_1.txt'

# Setting hyperparameters
learning_rate = 0.001
batch_size = 32
num_epoch = 50

# tensor([0.3778, 0.4980, 0.1993]) -> mean
# tensor([0.1722, 0.1499, 0.1384]) -> std

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Loading dateset (loading training, testing and validating sets)
train_set = StrawberryDataset(csv_file='dataset/data_train.csv', image_dir='dataset/train',
                              transform=transform)
test_set = StrawberryDataset(csv_file='dataset/data_test.csv', image_dir='dataset/test',
                             transform=transform)
valid_set = StrawberryDataset(csv_file='dataset/data_val.csv', image_dir='dataset/val',
                              transform=transform)

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
file = open(file_name, 'a')
n_total_steps = len(train_loader)
for epoch in range(num_epoch):
    train_loss = 0.0
    valid_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    model.eval()
    for image, label in valid_loader:
        image = image.to(device)
        label = label.to(device)
        output = model(image)
        loss = criterion(output, label)
        # update-average-validation-loss
        valid_loss += loss.item() * image.size(0)
    # calculate-average-losses
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    # print-training/validation-statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    file.write('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

# test-the-model
model.eval()  # it-disables-dropout
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model: {} %'.format(100 * correct / total))
    file.write('Test Accuracy of the model: {} % \n'.format(100 * correct / total))

# plotting the training and validation loss
plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.legend()
plt.show()
file.close()

