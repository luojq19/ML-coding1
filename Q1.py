# -*- codeing = utf-8 -*-
# @Time:  2:27 下午
# @Author: Jiaqi Luo
# @File: Q1.py.py
# @Software: PyCharm

# %%

import torch
import torchvision
from torch import nn
from torchvision import transforms

batch_size = 64

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.5], std=[0.5])])

data_train = torchvision.datasets.MNIST(root="./data/",
                                        transform=transform,
                                        train=True,
                                        download=True)

data_test = torchvision.datasets.MNIST(root="./data/",
                                       transform=transform,
                                       train=False)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=batch_size,
                                                shuffle=True)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=batch_size,
                                               shuffle=True)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(stride=2, kernel_size=2))
        self.fc = nn.Sequential(nn.Linear(14 * 14 * 128, 1024),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(1024, 10))

    def forward(self, x):
        return self.fc(self.conv(x).view(-1, 14 * 14 * 128))


lr = 0.001
num_epochs = 10

model = Model()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    # training
    sum_loss = 0.0
    train_correct = 0
    for data in data_loader_train:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        l = loss(outputs, labels)
        l.backward()
        optimizer.step()

        _, id = torch.max(outputs, 1)
        sum_loss += l.item()
        train_correct += torch.sum(id == labels).item()

    print('[%d,%d] loss:%.03f' % (epoch + 1, num_epochs, sum_loss / len(data_loader_train)))
    print('        correct:%.03f%%' % (100 * train_correct / len(data_train)))

# test
model.eval()
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in data_loader_test:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
torch.save(model.state_dict(), 'mnist_classifier.pth')
