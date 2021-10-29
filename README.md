import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os

import torch
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torch.nn as nn
from tqdm.notebook import tqdm
from torchsummary import summary
from PIL import Image

! mkdir ~/.kaggle
! cp kaggle.json  ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download   "drscarlat/melanoma"

! unzip -q "/content/melanoma.zip"

Загрузка данных:

data_image_train='/content/DermMel/train_sep'
data_image_test='/content/DermMel/test'
data_image_valid='/content/DermMel/valid'
resnet_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
train = datasets.ImageFolder(data_image_train, transform=resnet_transforms)
test = datasets.ImageFolder(data_image_test, transform=resnet_transforms)
valid = datasets.ImageFolder(data_image_valid, transform=resnet_transforms)

data_train = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
data_test = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)
data_valid = torch.utils.data.DataLoader(valid, batch_size=32, shuffle=False)

for batch in data_train:
    break

batch[0].shape, batch[1].shape

Пример данных:

dataiter = iter(data_train)
images, labels = dataiter.next()

def show_imgs(imgs, labels):
    f, axes= plt.subplots(1, 2, figsize=(15, 5))
    for i, axis in enumerate(axes):
        axes[i].imshow(np.squeeze(np.transpose(imgs[i].numpy(), (1, 2, 0))), cmap='gray')
        axes[i].set_title(labels[i].numpy())
    plt.show()

show_imgs(images, labels)

Функция обучения:

def train(model, n_epoch=6):
    loss_fn = torch.nn.CrossEntropyLoss()

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_accuracies = []
    val_losses = []
    
    for epoch in tqdm(range(n_epoch)):
        train_dataiter = iter(data_train)
        running_loss = 0.0

        model.train(True)
        for i, batch in enumerate(tqdm(train_dataiter)):
            X_batch, y_batch = batch
            
            logits = model(X_batch.to(device))
            loss = loss_fn(logits, y_batch.to(device)) 
            loss.backward() 
            optimizer.step() 
            optimizer.zero_grad() 
            running_loss += loss.detach().cpu().item()

            if i % 50 == 49:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 49))
                train_losses.append(running_loss / 49)
                running_loss = 0.0

        model.train(False)
        val_dataiter = iter(data_valid)


        val_loss_per_epoch = 0
        val_accuracy_per_epoch = 0
        for i, batch in enumerate(tqdm(val_dataiter)):
            X_batch, y_batch = batch
            with torch.no_grad():
                logits = model(X_batch)
                y_pred = torch.argmax(logits, dim=1)
                val_accuracy_per_epoch += np.mean(y_pred.numpy() == y_batch.numpy())

                val_loss_per_epoch += loss_fn(logits, y_batch)

        val_accuracies.append(val_accuracy_per_epoch / (i + 1))
        val_losses.append(val_loss_per_epoch / (i + 1))

    print('Обучение закончено')
    return model, train_losses, val_losses, val_accuracies
    
Посторение модели:

class ConvNet(nn.Module):
      def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,kernel_size=(7,7))
        nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(6,6), padding='same')
        nn.ReLU()
        self.bn0 = nn.BatchNorm2d(32)
        self.mpool = nn.MaxPool2d(kernel_size=(2,2))
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(6,6),padding='same')
        nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(5,5), padding='same')
        nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)
        self.apool = nn.AvgPool2d(kernel_size=(2, 2))
        self.conv5 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(5,5), padding='same')
        nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(5,5), padding='same')
        nn.ReLU()
        self.bn2 = nn.BatchNorm2d(128)
        self.apool2 = nn.AvgPool2d(kernel_size=(2, 2))
        self.conv7 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(4,4),padding='same')
        nn.ReLU()
        self.conv8 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(4,4), padding='same')
        nn.ReLU()
        self.bn3 = nn.BatchNorm2d(256)
        self.mpool2 = nn.MaxPool2d(kernel_size=(2,2))
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512,kernel_size=(3,3),padding='same')
        nn.ReLU()
        self.conv10 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),padding='valid')
        nn.ReLU()
        self.bn4 = nn.BatchNorm2d(512)
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(100 * 100 * 256, 256)
        nn.Sigmoid()
        self.fc2 = nn.Linear(256, 2)
      def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn0(x)
        x = self.mpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.bn1(x) 
        x = self.apool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.bn2(x)
        x = self.apool2(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.bn3(x)
        x = self.mpool2
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.bn4(x)
        x = self.flatten1(x)
        x = torch.sigmoid(self.fc1(x))
        x  = self.fc2(x)
          
        return x
        
        class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(7,7))
        nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(5,5))
        nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3))
        self.pool = nn.MaxPool2d(kernel_size=(2,2)) 
        nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3))
        self.bn1 = nn.BatchNorm2d(128)
        nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3)) 
        nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3)) 
        nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(100 * 100 * 256, 256)
        nn.Sigmoid()
        self.fc2 = nn.Linear(256, 2)

    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.bn1(x) 
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.flatten(x)
        x = torch.sigmoid(self.fc1(x))

        x  = self.fc2(x)
        
        return x
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

conv_net = ConvNet().to(device)
conv_net

data_train

for batch in data_train:
    break

batch[0].shape, batch[1].shape

conv_net, train_losses_conv, val_lossesmodel_conv, val_accuracies_conv = train(conv_net, n_epoch=1)

model=models.resnet18(pretrained=False)
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

model, train_losses_conv, val_lossesmodel_conv, val_accuracies_conv = train(model, n_epoch=1)

model, train_losses_conv, val_lossesmodel_conv, val_accuracies_conv = train(model, n_epoch=1)

torch.save(model.state_dict(), os.path.join('/content',"melanoma"))
torch.save(model.state_dict(), os.path.join('/content',"melanoma"))
