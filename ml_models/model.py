import torch
import torch.nn as nn
#CNNクラスの定義
class CNN(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=5,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        #サイズを1*1にする
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        #全結合層の設定(サイズ32がMaxPool2dを経て、サイズ8になっている)
        # self.classifier = nn.Linear(in_features=8 * 8 * 128,out_features=num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128,out_features=num_classes)
        )

    def forward(self,x):
        x = self.features(x)
        # x = x.view(x.size(0),-1)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x
