# @Time    : 2020/3/31 13:56
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : model.py
import numpy as np
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable
from torch.utils.data import DataLoader

from config import Config
from generate_num import GenetorNum, DealDataset


class GRUNet(nn.Module):
    def __init__(self, input_size=Config.NUM_CLASSES, hidden_size=32, out_dim=Config.NUM_CLASSES):
        super(GRUNet, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=Config.NUM_LAYERS,
            batch_first=True,
            bidirectional=False
        )
        self.fcLayer = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        r_out, _ = self.rnn(x)  # None 表示 hidden state 会用全0的 state
        r_out = F.leaky_relu(r_out)
        out = self.fcLayer(r_out)
        return out

def train():
    net = GRUNet(input_size=Config.NUM_CLASSES, hidden_size=Config.HIDDEN_SIZE, out_dim=Config.NUM_CLASSES)
    net.train()
    # 定义损失函数和优化函数
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)
    
    # 实例化这个类，然后我们就得到了Dataset类型的数据，记下来就将这个类传给DataLoader，就可以了。
    dealDataset = DealDataset()
    train_loader = DataLoader(dataset=dealDataset,
                              batch_size=32,
                              shuffle=True)
    for epoch in range(1, 50):
        print('epoch: %d' % (epoch,))
        for i, data in enumerate(train_loader):
            inputs, labels = data
            # 将这些数据转换成Variable类型
            inputs, labels = Variable(inputs).float(), Variable(labels).float()
            # forward
            output = net(inputs)
            output = output[:, -1]
            # output = output[:, -1,:]
            loss = criterion(output, labels)
            # update paramters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)
    
    print('train over')
    # 仅保存和加载模型参数(推荐使用)
    torch.save(net.state_dict(), Config.MODEL_PATH)


def eval():
    result = list()
    net = GRUNet(input_size=Config.NUM_CLASSES, hidden_size=Config.HIDDEN_SIZE, out_dim=Config.NUM_CLASSES)
    net.load_state_dict(torch.load(Config.MODEL_PATH))
    net.eval()
    dealDataset = DealDataset()
    train_loader = DataLoader(dataset=dealDataset,
                              batch_size=64,
                              shuffle=False)
    for i, data in enumerate(train_loader):
        inputs, labels = data
        x = Variable(torch.tensor(inputs).float())
        out = net(x)[:, -1]
        out = out.detach().numpy()
        out = np.argsort(out, axis=-1)
        # out = np.argmax(out.detach().numpy(), axis=-1)
        for i in range(inputs.shape[0]):
            print(np.argmax(inputs[i], axis=-1), np.argmax(labels[i], axis=-1), out[i][::-1])
    
    # x = Variable(torch.tensor(x).float())
    # for i in range(100):
    #     # data_x = GenetorNum.random_num(shape=Config.DATA_SHAPE)
    #     # x = torch.from_numpy(GenetorNum.data_one_hot(data_x, num_classes=Config.NUM_CLASSES)[0:1]).float()
    #     out = net(x)
    #     # pred = out.detach().numpy()
    #     # x_ = np.argmax(x, axis=2)
    #     # y_ = np.argmax(pred, axis=2)
    #     # print('x: {}, pred: {}'.format(x_, y_))
    #     out = np.argmax(out.detach().numpy(), axis=2)
    #     out = out[0,-1, 0]
    #
    #     result.append(out)
    #     x[0,:-1,0] = x[0, 1:, 0]
    #     x[0, -1, 0] = out
    #
    # fig, ax = plt.subplots()
    # ax.plot(range(len(result)), result, color='b')
    # plt.show()
def predict():
    net = GRUNet(input_size=Config.NUM_CLASSES, hidden_size=Config.HIDDEN_SIZE, out_dim=Config.NUM_CLASSES)
    net.load_state_dict(torch.load(Config.MODEL_PATH))
    net.eval()

    inputs = GenetorNum.read_pailie3_last()
    x = Variable(torch.tensor(inputs).float())
    out = net(x)[:, -1]
    out = out.detach().numpy()
    out = np.argsort(out, axis=-1)
    # out = np.argmax(out.detach().numpy(), axis=-1)
    for i in range(inputs.shape[0]):
        print('输入值：{}， 预测值：{}'.format(np.argmax(inputs[i], axis=-1), out[i][::-1]))

if __name__ == '__main__':
    train()
    eval()
    # predict()
   