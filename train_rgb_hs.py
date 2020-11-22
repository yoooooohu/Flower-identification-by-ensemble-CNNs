# encoding: utf-8

# from se_module import SELayer
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import time
import scipy.io as sio
import numpy as np
import csv
import time

import torch.nn.functional as F

start_time = time.perf_counter()

torch.cuda.set_device(5)  # use the chosen gpu

root = '/home/huyoupeng/flower_classification/'  # 工程路径
num_epochs = 900  # 要求num_each_epoch的倍数 default = 900
num_each_epoch = 5
batch_size = 128  # default = 32 大batch size在显存能允许的情况下收敛速度是比较快的但有时的确会有陷入局部最小的情况
# 小batch size引入的随机性会更大些，有时候能有更好的效果，但是就是收敛速度慢一些
momentum_num = 0.9
learning_rate = 1e-3
correct = 0
total = 0

img_transforms = transforms.Compose([
    transforms.Resize(223),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(223),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_transforms = transforms.Compose([
    transforms.Resize(223),
    transforms.CenterCrop(223),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

img_transforms_2 = transforms.Compose([
    transforms.Resize(400),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def get_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))]


def default_loader(path):
    return Image.open(path).convert('RGB'), Image.open(path).convert('HSV')
    # return sio.loadmat(path)


def Visualize_image(path):
    img = Image.open(path).convert('RGB')
    img = img_transforms(img)
    img2 = transforms.ToPILImage()(img).convert('RGB')
    img2.show()


class MyDataset(Dataset):
    def __init__(self, txt, transform=img_transforms, target_transform=img_transforms, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split('   ')
            imgs.append((root + 'segmim/' + words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img_rgb, img_hsv = self.loader(fn)
        if self.transform is not None:
            img_rgb = self.transform(img_rgb)
            img_hsv = self.transform(img_hsv)
            # img = torch.cat([img_rgb, img_hsv[0:2]], dim=0)


        return img_rgb, img_hsv, label

    def __len__(self):
        return len(self.imgs)


train_data = MyDataset(txt='/home/huyoupeng/flower_classification/Flower-identification-by-ensemble-CNNs/train.txt', transform=img_transforms)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data=MyDataset(txt='/home/huyoupeng/flower_classification/Flower-identification-by-ensemble-CNNs/test.txt', transform=test_transforms)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


class AlexNet(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=11, stride=4, padding=2),   # 64  223 223

            nn.BatchNorm2d(64, momentum=momentum_num),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),         #64  55  55
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  #192 27 27

            nn.BatchNorm2d(192, momentum=momentum_num),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),    #192  27 27
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  #384 13 13

            nn.BatchNorm2d(384, momentum=momentum_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  #256 13 13

            nn.BatchNorm2d(256, momentum=momentum_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  #256 13 13

            nn.BatchNorm2d(256, momentum=momentum_num),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), )    #256 13 13
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes), ######################################
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.num_classes = num_classes
        self.Alexnet_rgb = AlexNet(3, num_classes)
        self.Alexnet_hs = AlexNet(2, num_classes)

        self.classifier = nn.Linear(4096*2, num_classes)


    def forward(self, x0, x1):
        x_rgb = self.Alexnet_rgb(x0)
        x_hs = self.Alexnet_rgb(x1)
        x = torch.cat([x_rgb, x_hs], dim=1)
        # print(x.shape)
        x = self.classifier(x)

        return x


def train_alexnet():
    model = MyModel(102)
    model.cuda()
    model.train()
    now = time.strftime('%Y-%m-%d-%H-%M-%S')

    for cnt in range(round(num_epochs / num_each_epoch)):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(num_each_epoch):
            print('epoch {}'.format(epoch + 1 + cnt * num_each_epoch))
            for i, (batch_x0_original, batch_x1_original, batch_y_original) in enumerate(train_loader):
                batch_x0 = Variable(batch_x0_original).cuda()
                batch_x1 = Variable(batch_x1_original).cuda()
                batch_y = Variable(batch_y_original).cuda()
                optimizer.zero_grad()
                out = model(batch_x0, batch_x1)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                _, correct_label = torch.max(out, 1)  # 输出预测概率最大的值和标签
                correct_num = (correct_label == batch_y).sum()
                acc = correct_num.item() / batch_y.size(0)  # 一轮个数
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, Acc: %.4f' %
                      (epoch + 1 + cnt * num_each_epoch, num_epochs, i + 1,
                       len(train_data) // batch_size, loss.item(), acc))

            model.eval()
            correct = 0
            total = 0
            for batch_x0, batch_x1, batch_y in test_loader:
                batch_x0= Variable(batch_x0.cuda())
                batch_x1= Variable(batch_x1.cuda())
                out = model(batch_x0, batch_x1)
                _, predicted = torch.max(out.data, 1)
                total += batch_y.size(0)
                correct += (predicted.cpu() == batch_y).sum()

            corrext_rate = correct.item() / total
            print('Test Accuracy of the model on the %d test images: %.4f' %(total, corrext_rate ))
            model.train()

            f = open('/home/huyoupeng/flower_classification/Flower-identification-by-ensemble-CNNs/result/{}record_rgb_hs.txt'.format(now), 'a', encoding='utf-8', newline='')
            csv_writer = csv.writer(f)
            csv_writer.writerow([str(epoch + 1 + cnt * num_each_epoch), str(corrext_rate)])
            torch.save(model.state_dict(), './checkpoints/rgb_hs_alexnet.pth')  # 一个epoch保存一次模型
    end_time = time.perf_counter()
    print("Running time: ", (end_time - start_time) / 60)


if __name__ == '__main__':
    train_alexnet()
