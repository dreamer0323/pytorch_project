# 定义模型
import torch

#继承 torch.nn.Module
class model(torch.nn.Module):

    #定义模型参数
    def __init__(self):
        super().__init__()
        #使用序列化容器管理模型参数
        self.layers = torch.nn.Sequential(
            #采用CNN网络为主要结构
            torch.nn.Conv2d(1,64,3,stride=1,padding=1), #卷积层1
            torch.nn.MaxPool2d(2,2), #最大池化层1
            torch.nn.ReLU(), #非线性激活层数 1

            torch.nn.Conv2d(64,32,3,stride=1,padding=1), #卷积层2
            torch.nn.MaxPool2d(2,2), #最大池化层2
            torch.nn.ReLU(), #非线性激活层数 2

            torch.nn.Conv2d(32,16,3,stride=1,padding=1), #卷积层3
            torch.nn.MaxPool2d(2,2,padding = 1), #最大池化层3
            torch.nn.ReLU(), #非线性激活层数3

            torch.nn.Flatten(), #展平层
            torch.nn.Linear(16*4*4,10),  #全连接层1
        )

    def forward(self,x):
        return self.layers(x)


if __name__ == "__main__":
    # 打印模型参数
    print(model())
