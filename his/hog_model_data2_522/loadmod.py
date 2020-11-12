import torch
import torch.nn as nn


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


model = torch.load('900_0.3314470720720721_flower_model_hog.pkl')