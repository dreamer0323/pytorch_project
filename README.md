# PyTorch 手写数字识别（MNIST）

基于 **PyTorch + CNN（卷积神经网络）** 的 MNIST 手写数字分类项目。支持训练、测试、断点续训、早停，以及 TensorBoard 训练可视化。

## 项目简介

使用一个简单的 3 层卷积神经网络对 MNIST 手写数字数据集（0-9 共 10 类）进行分类，测试集最佳准确率达到 **98.77%**。

## 项目结构

```
Mnist.py          程序入口，通过参数切换 train / test 模式
model.py          模型定义（3 层卷积 + 全连接）
train.py          训练逻辑（损失、优化、早停、断点续训、TensorBoard）
test.py           加载训练好的权重，评估测试集准确率
common.py         数据加载 + 评估函数
param.py          超参数类
best_model.pth    最佳模型权重
checkpoint.pth    最近一次检查点（可断点续训）
data/             MNIST 数据集
runs/             TensorBoard 日志
```

## 模型结构

| 层 | 类型 | 说明 |
| --- | --- | --- |
| 1 | Conv2d(1, 64, 3) + MaxPool2d(2) + ReLU | 卷积层 1 |
| 2 | Conv2d(64, 32, 3) + MaxPool2d(2) + ReLU | 卷积层 2 |
| 3 | Conv2d(32, 16, 3) + MaxPool2d(2) + ReLU | 卷积层 3 |
| 4 | Flatten + Linear(16×4×4, 10) | 展平 + 全连接分类 |

- **损失函数**：交叉熵 `CrossEntropyLoss`
- **优化器**：`SGD`（学习率 0.01）
- **设备**：自动检测 CUDA，无 GPU 时回退 CPU

## 环境依赖

```bash
pip install torch torchvision tensorboard
```

## 使用方法

```bash
# 训练模型
python Mnist.py train

# 测试模型（加载 best_model.pth 评估准确率）
python Mnist.py test

# 查看训练曲线
tensorboard --logdir=runs
```

## 超参数设置

在 `Mnist.py` 中调整：

```python
p = param(batch_size=100, learning_rate=0.01, epochs=600)
```

在 `train.py` 中调整早停策略：

```python
patience = 10   # 连续多少个 epoch 准确率无提升则提前停止
```

## 训练特性

- **断点续训**：每个 epoch 保存 `checkpoint.pth`，中断后再次运行会自动从上次进度继续；手动 `Ctrl+C` 中断也会保存当前进度。
- **早停**：连续 `patience` 轮测试准确率无提升时自动停止训练。
- **最佳模型保存**：准确率提升时自动保存为 `best_model.pth`。
- **可视化**：训练损失和测试准确率记录到 TensorBoard。

## 训练结果

| 指标 | 值 |
| --- | --- |
| 最佳测试集准确率 | **98.77%** |

## 许可证

个人学习项目，可自由使用。
