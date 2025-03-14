import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import torchvision.datasets as datasets
transform_train = transforms.Compose(
    [transforms.Pad(4),

     transforms.Resize(224),  # 将图像大小调整为(224, 224
     transforms.ToTensor(),
     # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     transforms.RandomHorizontalFlip(),
     transforms.RandomGrayscale(),
     # transforms.RandomCrop(32, padding=4),
     ])

transform_test = transforms.Compose(
    [transforms.Pad(4),

     transforms.Resize(224),  # 将图像大小调整为(224, 224
     transforms.ToTensor(),
     # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     transforms.RandomHorizontalFlip(),
     transforms.RandomGrayscale(),
     # transforms.RandomCrop(32, padding=4),
     ]
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# trainset = torchvision.datasets.CIFAR10(root='dataset_method_1', train=True, download=True, transform=transform_train)
# trainLoader = torch.utils.data.DataLoader(trainset, batch_size=24, shuffle=True)
#
# testset = torchvision.datasets.CIFAR10(root='dataset_method_1', train=False, download=True, transform=transform_test)
# testLoader = torch.utils.data.DataLoader(testset, batch_size=24, shuffle=False)

trainset = datasets.MNIST(root='./data', train=True, transform=transform_train, download=True)

testset = datasets.MNIST(root='./data', train=False, transform=transform_test)

train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=16, shuffle=False)


vgg = [96, 96, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']




class VGG(nn.Module):
    def __init__(self, vgg):
        super(VGG, self).__init__()
        self.features = self._make_layers(vgg)
        self.dense = nn.Sequential(
            nn.Linear(512*7*7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        out = self.features(x)
        # out = out.view(out.size(0), -1)

        out = torch.flatten(out, 1)
        out = self.dense(out)
        out = self.classifier(out)
        return out

    def _make_layers(self, vgg):
        layers = []
        in_channels = 1
        for x in vgg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


model = VGG(vgg)
# model.load_state_dict(torch.load('CIFAR-model/VGG16.pth'))


# print(model)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-3)
loss_func = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.4, last_epoch=-1)

total_times = 40
total = 0
accuracy_rate = []


def test():
    model.eval()
    correct = 0  # 预测正确的图片数
    total = 0  # 总共的图片数
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            outputs = model(images).to(device)
            outputs = outputs.cpu()
            outputarr = outputs.numpy()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
    accuracy = 100 * correct / total
    accuracy_rate.append(accuracy)
    print(f'准确率为:{accuracy}%'.format(accuracy))


for epoch in range(total_times):
    model.train()
    model.to(device)
    running_loss = 0.0
    total_correct = 0
    total_trainset = 0

    for i, (data, labels) in enumerate(train_loader, 0):
        data = data.to(device)
        outputs = model(data).to(device)
        labels = labels.to(device)
        loss = loss_func(outputs, labels).to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = outputs.max(1)
        correct = (pred == labels).sum().item()
        total_correct += correct
        total_trainset += data.shape[0]
        if i % 1000 == 0 and i > 0:
            print(f"正在进行第{i}次训练, running_loss={running_loss}".format(i, running_loss))
            running_loss = 0.0
    test()
    scheduler.step()

# torch.save(model.state_dict(), 'CIFAR-model/VGG16.pth')
accuracy_rate = np.array(accuracy_rate)
times = np.linspace(1, total_times, total_times)
plt.xlabel('times')
plt.ylabel('accuracy rate')
plt.plot(times, accuracy_rate)
plt.show()

print(accuracy_rate)