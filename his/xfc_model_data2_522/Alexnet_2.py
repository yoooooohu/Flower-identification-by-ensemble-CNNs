#------------------HOG版本------------------

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dsets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from skimage import io
import scipy.io as sio
import visdom		# python -m visdom.server
import time

#------------------parameter------------------
root='D:/Project_File/py_Project/Alexnet/'		#工程路径
model_name = 'flower_model_hog.pkl'			#模型名称
#train_test = 0 		#0为train 1为test

num_epochs = 900		#要求num_each_epoch的倍数
num_each_epoch = 5
batch_size = 64		#大batch size在显存能允许的情况下收敛速度是比较快的但有时的确会有陷入局部最小的情况
					#小batch size引入的随机性会更大些，有时候能有更好的效果，但是就是收敛速度慢一些

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
    transforms.Resize(28),
    transforms.CenterCrop(28),
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

# #读取文件
def get_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))]

# -----------------ready the dataset--------------------------
def default_loader(path):
    # return Image.open(path).convert('RGB')
    return sio.loadmat(path)
#
def Visualize_image(path):
    img=Image.open(path).convert('RGB')
    img=img_transforms(img)
    img2 = transforms.ToPILImage()(img).convert('RGB')
    img2.show()

# Visualize_image('/home/hbz/PycharmProjects/test_flower/image_00001.jpg')
# Visualize_image('/home/hbz/PycharmProjects/test_flower/train/image_0002.jpg')

class MyDataset(Dataset):
    def __init__(self, txt, transform=img_transforms, target_transform=img_transforms, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((root + words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)['reshape_hog']	#装进来是dict 提取变量
        img = np.array(img)			#转np的array   36*29*29
        img = torch.from_numpy(img)
        #if self.transform is not None:
        #    img = self.transform(img)
        return img,label
    def __len__(self):
        return len(self.imgs)

############################ main ####################################
train_data=MyDataset(txt=root+'train1_2.txt', transform=img_transforms)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
#用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照
#	batch size封装成Tensor，后续只需要再包装成Variable即可作为模型的输入

test_data=MyDataset(txt=root+'test1_2.txt', transform=img_transforms)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# #-----------------create the Net and training------------------------
#输入几类，输出几类有影响，注意修改
class AlexNet_hog(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet_hog, self).__init__()
        self.features = nn.Sequential(
        	# 输入36通道，输出64通道，卷积核9*9,卷积核的步长3,扩充边缘2
            nn.Conv2d(36 , 64, kernel_size=9, stride=3, padding=2),   # 36 29 29->64  9  9
            										#(n+2*padding-kernel_size)/stride+1
            nn.BatchNorm2d(64, momentum=momentum_num),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=1),		
            nn.Conv2d(64, 192, kernel_size=3, padding=1),		# 64 9 9->192 9  9

            nn.BatchNorm2d(192, momentum=momentum_num),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=1),    		
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  	# 192 9 9->384 9  9

            nn.BatchNorm2d(384, momentum=momentum_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  #256 9 9

            nn.BatchNorm2d(256, momentum=momentum_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  #256 9 9

            nn.BatchNorm2d(256, momentum=momentum_num),
            nn.ReLU(inplace=True),
            #nn.Conv2d(256, 256, kernel_size=7, padding=1),  #256 5 5
            nn.MaxPool2d(kernel_size=3, stride=2),     	#256 4 4
            )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # 定义全连接层：线性连接(y = Wx + b)，256*3*3个节点连接到1024个节点上
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes), 
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 4 * 4)	      #尺度变换（256,2,2)->(256*2*2,0,0)
        x = self.classifier(x)
        return x

vis = visdom.Visdom(env=u'test1')
vis.line(X=torch.Tensor([0]),Y=torch.Tensor([0]),win='acc',opts={'title': 'acc rate'})

# if os.path.exists(model_name):		#存在flower_model.pkl
# 	model = torch.load(model_name);
# else:
model = AlexNet_hog(102)
if torch.cuda.is_available():
	model.cuda()
time_start = time.time()

for cnt in range(round(num_epochs/num_each_epoch)):
	#------------------net---------------------------------
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	for epoch in range(num_each_epoch):
	    print('epoch {}'.format(epoch + 1 + cnt*num_each_epoch))
	    train_loss = 0.
	    train_acc = 0.
	    for i,(batch_x, batch_y) in enumerate(train_loader):
	        if torch.cuda.is_available():
	            batch_x = Variable(batch_x).cuda()
	            batch_y = Variable(batch_y).cuda()
	        else:
	            batch_x = Variable(batch_x)
	            batch_y = Variable(batch_y)
	        optimizer.zero_grad()
	        out = model(batch_x)
	        # print('输出：', out)

	        loss = criterion(out, batch_y)
	        loss.backward()
	        optimizer.step()
	        if (i + 1) % 10 == 0:
	            _, correct_label = torch.max(out, 1)  # 输出预测概率最大的值和标签
	            # print('_:', _)
	            # print('label', correct_label)
	            correct_num = (correct_label == batch_y).sum()
	            acc = correct_num.item() / batch_y.size(0)	#一轮个数

	            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, Acc: %.4f' %
	                  (epoch + 1 + cnt*num_each_epoch, num_epochs, i + 1,
	                   len(train_data) // batch_size, loss.item() , acc))

	    time_end = time.time()
	    print(str(time_end - time_start)+'/ epoch')
	#torch.save(model, root + 'ModelBackup/epoch' + str(epoch + 1 + cnt*num_each_epoch) + model_name)
	#------------------- evaluation --------------------------------
	#model=torch.load(root + 'ModelBackup/epoch' + str(epoch + 1 + cnt*num_each_epoch) + model_name)
	#print(model)
	model.eval()
	for batch_x, batch_y in test_loader:
	    batch_x= Variable(batch_x.cuda())
	    out = model(batch_x)
	    #print(out)
	    _, predicted = torch.max(out.data, 1)
	    #print('_:', _)
	    #print('predicted', predicted)
	    total += batch_y.size(0)
	    #print('总数:', total)
	    #print('预测类别：', predicted, '标签：', batch_y)
	    correct += (predicted.cpu() == batch_y).sum()
	    #print('正确：', correct)
	corrext_rate = correct.item() / total
	print('Test Accuracy of the model on the %d test images: %.4f' %(total, corrext_rate ))
	vis.line(X=torch.Tensor([cnt*num_each_epoch]),Y=torch.Tensor([corrext_rate*1000]),\
											win='acc',update='append',opts={'title': 'acc rate'})
	model.train()
	torch.save(model, root + 'ModelBackup/' + str(epoch + 1 + cnt*num_each_epoch) \
											+ '_' + str(corrext_rate) + '_' + model_name)