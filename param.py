import torch

# 定义参数类
class param:
    def __init__(self,batch_size,learning_rate=0.01,epochs=10):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #判断是否有cuda设备
        

