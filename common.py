"""
    公共函数：
    - 加载 MNIST 数据集
    - 在测试集上评估模型准确率（训练和测试两个模块共用）
"""
import torch
import torchvision
from torchvision import transforms
from torch.utils import data


def get_dataloaders(batch_size):
    """加载 MNIST 数据集，返回 (训练数据加载器, 测试数据加载器)"""
    # 训练集 转换为张量Tensor
    dataset = torchvision.datasets.MNIST(root="./data",train=True,download=True,transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(root="./data",train=False,download=True,transform=transforms.ToTensor())

    # 训练集打乱，测试集无需打乱
    dataloader = data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
    test_dataloader = data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    return dataloader, test_dataloader


def evaluate(model, test_dataloader, device):
    """在测试集上评估模型，返回整个测试集的准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(): #禁用梯度计算
        for image,label in test_dataloader:
            #将数据移动到指定设备
            image = image.to(device)
            label = label.to(device)
            # 前向传播
            output = model(image)
            predict = torch.argmax(output,dim=1)
            correct += (predict == label).sum().item()
            total += len(label)
    return correct / total
