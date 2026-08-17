"""
    测试模块：加载训练好的模型参数，在测试集上评估准确率
"""
import os
import torch
from model import model
from common import get_dataloaders, evaluate


def test(p):
    _, test_dataloader = get_dataloaders(p.batch_size)

    # 导入神经网络模型 并将模型移动到指定设备
    my_model = model().to(p.device)
    # 优先加载最佳模型，其次加载最后一次检查点
    if os.path.exists("best_model.pth"):
        # weights_only=True：只加载张量等安全类型，拒绝反序列化任意对象，防止恶意 .pth 文件在加载时执行代码
        state_dict = torch.load("best_model.pth", map_location=p.device, weights_only=True)
        print("加载最佳模型 best_model.pth")
    elif os.path.exists("checkpoint.pth"):
        state_dict = torch.load("checkpoint.pth", map_location=p.device, weights_only=True)["model_state_dict"]
        print("加载最后一次检查点 checkpoint.pth")
    else:
        print("没有找到模型参数文件（best_model.pth / checkpoint.pth），请先运行训练：python Mnist.py train")
        return

    # 加载模型参数
    my_model.load_state_dict(state_dict)

    # 评估模型
    accuracy = evaluate(my_model, test_dataloader, p.device)
    print(f"测试集准确率:{accuracy}")
