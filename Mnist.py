"""
    程序入口：通过 mode 变量判断当前是训练还是测试
        训练：python Mnist.py train
        测试：python Mnist.py test
"""
import sys
from param import param


if __name__ == "__main__":
    # 超参数设置
    p = param(batch_size=100, learning_rate=0.01, epochs=600)

    # 用一个变量判断当前是训练还是测试（默认 train）
    #   train -> 训练模型
    #   test  -> 加载已训练参数并评估

    #等价于三元表达式 mode = sys.argv[1] if len(sys.argv) > 1 else "train" # 默认 train 模式 ，用户可以指定 test 模式 
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "train"

    
    if mode == "train":
        from train import train
        train(p)
    elif mode == "test":
        from test import test
        test(p)
    else:
        print("mode 参数错误，只能是 train 或 test")
