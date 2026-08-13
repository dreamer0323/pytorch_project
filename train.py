"""
    训练模块：负责模型训练、TensorBoard 记录、检查点保存和提前停止
"""
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from model import model
from common import get_dataloaders, evaluate


def train(p):
    # 加载数据
    dataloader, test_dataloader = get_dataloaders(p.batch_size)

    # 导入神经网络模型 并将模型移动到指定设备
    my_model = model().to(p.device)
    # 定义损失函数 使用交叉熵损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 定义优化器 使用SGD优化器
    optimizer = torch.optim.SGD(my_model.parameters(),lr=p.learning_rate)

    # 提前停止相关设置
    patience = 10            # 连续多少个epoch准确率没有提升就提前停止
    best_accuracy = 0.0      # 记录历史最佳准确率
    epochs_no_improve = 0    # 连续没有提升的epoch计数

    # tensorboard 记录器，训练结束后在命令行执行 tensorboard --logdir=runs 查看
    writer = SummaryWriter("runs/mnist")

    # 如果存在上次中断保存的检查点，则加载后继续训练
    start_epoch = 0
    if os.path.exists("checkpoint.pth"):
        checkpoint = torch.load("checkpoint.pth", map_location=p.device)
        my_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint.get("accuracy", 0.0)
        print(f"检测到检查点，从第{start_epoch+1}轮继续训练，历史最佳准确率:{best_accuracy}")

    # 训练模型
    epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, p.epochs):
            loss_total = 0
            my_model.train() #设置模型为训练模式

            for image,label in dataloader:
                #将数据移动到指定设备
                image = image.to(p.device)
                label = label.to(p.device)
                # 清空梯度
                optimizer.zero_grad()
                # 前向传播
                output = my_model(image)
                # 计算损失
                loss = loss_fn(output,label)
                loss_total += loss.item()
                # 反向传播
                loss.backward()
                # 更新参数
                optimizer.step()

            # 一整轮训练结束后，打印并记录本轮的真正平均损失
            avg_loss = loss_total/len(dataloader)
            print(f"第{epoch+1}轮训练，平均损失值:{avg_loss}")
            writer.add_scalar("Loss/train", avg_loss, epoch)

            # 每个epoch在测试集上评估一次准确率
            accuracy = evaluate(my_model, test_dataloader, p.device)
            print(f"第{epoch+1}轮测试集准确率:{accuracy}")
            writer.add_scalar("Acc/test", accuracy, epoch)

            # 准确率提升时保存最佳模型
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                epochs_no_improve = 0
                torch.save(my_model.state_dict(), "best_model.pth")
                print("  -> 保存当前最佳模型 best_model.pth")
            else:
                epochs_no_improve += 1

            # 每个epoch保存一次检查点，即使中断也能从这一轮恢复
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": my_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": best_accuracy,
            }, "checkpoint.pth")

            # 提前停止
            if epochs_no_improve >= patience:
                print(f"连续{patience}轮准确率没有提升，提前停止训练")
                break

    except KeyboardInterrupt:
        # 手动 Ctrl+C 中断时，保存当前进度，参数不会丢失
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": my_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "accuracy": best_accuracy,
        }, "checkpoint.pth")
        print("\n已手动中断，当前参数已保存到 checkpoint.pth")

    writer.close()
    print(f"训练结束，最佳测试集准确率:{best_accuracy}")
