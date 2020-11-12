import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dsets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import scipy.io
from skimage import io
from skimage import io
import scipy.io as sio
root=''
model_name = 'flower_model_raw.pkl'

num_epochs = 1200
num_each_epoch = 50
batch_size = 128
momentum_num = 0.7
learning_rate = 1e-3
correct = 0
total = 0

img_transforms = transforms.Compose([
    transforms.Scale(223),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(223),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_transforms = transforms.Compose([
    transforms.Scale(223),
    transforms.CenterCrop(223),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def get_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))]

def default_loader1(path):
    return Image.open(path).convert('L')


class MyDataset(Dataset):
    def __init__(self, txt, transform=img_transforms, target_transform=img_transforms, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label
    def __len__(self):
        return len(self.imgs)


train_data=MyDataset(txt=root+'train1.txt', transform=img_transforms, loader=default_loader1)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data=MyDataset(txt=root+'test1.txt', transform=test_transforms, loader=default_loader1)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)



class dwt_Alexnet(nn.Module):
    def __init__(self, num_classes):
        super(dwt_Alexnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),   # 64  223 223

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
            nn.Linear(4096, num_classes), )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

model = AlexNet(102)
if torch.cuda.is_available():
    model.cuda()

#    #------------------net---------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#

# for cnt in range(round(num_epochs/num_each_epoch)):
#     for epoch in range(num_each_epoch):
#         print('epoch {}'.format(epoch + 1 + cnt*num_each_epoch))
#         for i,(batch_x, batch_y) in enumerate(train_loader):
#             if torch.cuda.is_available():
#                 batch_x = Variable(batch_x).cuda()
#                 batch_y = Variable(batch_y).cuda()
#             else:
#                 batch_x = Variable(batch_x)
#                 batch_y = Variable(batch_y)
#             optimizer.zero_grad()
#             out = model(batch_x)
#             loss = criterion(out, batch_y)
#             loss.backward()
#             optimizer.step()
#             if (i + 1) % 10 == 0:
#                 _, correct_label = torch.max(out, 1)  # 输出预测概率最大的值和标签
#                 correct_num = (correct_label == batch_y).sum()
#                 acc = correct_num.data[0] / batch_y.size(0)	#一轮个数
#
#                 print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, Acc: %.4f' %
# 	                  (epoch + 1, num_epochs, i + 1,
# 	                   len(train_data) // batch_size, loss.data[0] , acc))
#
#     torch.save(model, root + 'epoch' + str(epoch + 1 + cnt*num_each_epoch) + model_name)
#
#     #------------------- evaluation --------------------------------
#     net=torch.load(root + 'epoch' + str(epoch + 1 + cnt*num_each_epoch) + model_name)
#     #print(net)
#     net.eval()
#     for batch_x, batch_y in test_loader:
#         batch_x= Variable(batch_x.cuda())
#         out = net(batch_x)
#         #print(out)
#         _, predicted = torch.max(out.data, 1)
#         #print('_:', _)
#         #print('predicted', predicted)
#         total += batch_y.size(0)
#         #print('总数:', total)
# 	    #print('预测类别：', predicted, '标签：', batch_y)
#         correct += (predicted.cpu() == batch_y).sum()
#         #print('正确：', correct)
#     corrext_rate = correct / total
#     print('Test Accuracy of the model on the %d test images: %.4f' %(total, corrext_rate ))


