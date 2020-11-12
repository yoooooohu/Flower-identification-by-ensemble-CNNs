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
root='/home/hbz/PycharmProjects/test_flower/'

num_epochs = 1000
batch_size = 128
momentum_num = 0.9
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
    transforms.Scale(28),
    transforms.CenterCrop(28),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

img_transforms_2 = transforms.Compose([
    transforms.Scale(400),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# #读取文件
def get_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))]

# -----------------ready the dataset--------------------------
def default_loader(path):
    return Image.open(path).convert('RGB')
#
def Visualize_image(path):
    img=Image.open(path).convert('RGB')
    img=img_transforms(img)
    img2 = transforms.ToPILImage()(img).convert('RGB')
    img2.show()


# Visualize_image('/home/hbz/PycharmProjects/test_flower/image_00001.jpg')
# # Visualize_image('/home/hbz/PycharmProjects/test_flower/train/image_0002.jpg')

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

train_data=MyDataset(txt=root+'train1.txt', transform=img_transforms)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data=MyDataset(txt=root+'test1.txt', transform=img_transforms)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# #-----------------create the Net and training------------------------
#输入几类，输出几类有影响，注意修改
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),   # 64  223 223

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
# for epoch in range(num_epochs):
#     print('epoch {}'.format(epoch + 1))
#     train_loss = 0.
#     train_acc = 0.
#     for i,(batch_x, batch_y) in enumerate(train_loader):
#         if torch.cuda.is_available():
#             batch_x = Variable(batch_x).cuda()
#             batch_y = Variable(batch_y).cuda()
#         else:
#             batch_x = Variable(batch_x)
#             batch_y = Variable(batch_y)
#         optimizer.zero_grad()
#         out = model(batch_x)
#         # print('输出：', out)
#
#         loss = criterion(out, batch_y)
#         loss.backward()
#         optimizer.step()
#         if (i + 1) % 10 == 0:
#             _, correct_label = torch.max(out, 1)  # 输出预测概率最大的值和标签
#             # print('_:', _)
#             # print('label', correct_label)
#             correct_num = (correct_label == batch_y).sum()
#             acc = correct_num.data[0] / batch_y.size(0)
#
#             print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, Acc: %.4f' %
#                   (epoch + 1, num_epochs, i + 1,
#                    len(train_data) // batch_size, loss.data[0], acc))
#
# torch.save(model, 'flower_model.pkl')




#  ########### evaluation #############################--------------------------------
net=torch.load('flower_model.pkl')
#print(net)
net.eval()

for batch_x, batch_y in test_loader:
    batch_x= Variable(batch_x.cuda())
    out = net(batch_x)
    #print(out)
    _, predicted = torch.max(out.data, 1)
    #print('_:', _)
    #print('predicted', predicted)
    total += batch_y.size(0)
    #print('总数:', total)
    #print('预测类别：', predicted, '标签：', batch_y)
    correct += (predicted.cpu() == batch_y).sum()
    #print('正确：', correct)

print('Test Accuracy of the model on the %d test images: %.4f' %(total, correct / total))