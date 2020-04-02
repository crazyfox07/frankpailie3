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

num_classes = 10


class Model(object):
    def __init__(self):
        self.df = GenetorNum.sin_num()
    
    def random_forest_regressor(self):
        x0, y = self.df['B'], self.df['D']
        
        x0 = np.reshape(x0.values, newshape=(x0.shape[0], 1))
        x_train, x_test = x0[:800], x0[800:]
        y_train, y_test = y[:800], y[800:]
        print(x0.shape, y.shape)
        est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                        max_depth=3, random_state=0, loss='ls').fit(x_train, y_train)
        y_pred = est.predict(x_test)
        print('x_test.shape: {},y_pred.shape: {},y_test.shape: {}'.format(x_test.shape, y_pred.shape, y_test.shape))
        fig, ax = plt.subplots()
        ax.scatter(x_test, y_pred, marker='<', color='r', linewidth=2, label='y_pred')
        ax.scatter(x_test, y_test, marker='*', color='g', linewidth=2, label='y_test')
        plt.show()
        loss = mean_squared_error(y_test, est.predict(x_test))
        print('loss: {}'.format(loss))


class GRUNet(nn.Module):
    
    def __init__(self, input_size=Config.NUM_CLASSES, hidden_size=64, out_dim=Config.NUM_CLASSES):
        super(GRUNet, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.fcLayer = nn.Linear(hidden_size, out_dim)
    
    def forward(self, x):
        r_out, _ = self.rnn(x)  # None 表示 hidden state 会用全0的 state
        x = F.relu(r_out)
        out = self.fcLayer(x)
        return out


def train():
    net = GRUNet(input_size=1, hidden_size=64, out_dim=1)
    net.train()
    # 定义损失函数和优化函数
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    
    # 实例化这个类，然后我们就得到了Dataset类型的数据，记下来就将这个类传给DataLoader，就可以了。
    dealDataset = DealDataset()
    train_loader = DataLoader(dataset=dealDataset,
                              batch_size=64,
                              shuffle=True)
    for epoch in range(1, 10):
        print('epoch: %d' % (epoch,))
        for i, data in enumerate(train_loader):
            inputs, labels = data
            # 将这些数据转换成Variable类型
            inputs, labels = Variable(inputs).float(), Variable(labels).float()
            # forward
            output = net(inputs)
            output = output[:, -1, :]
            loss = criterion(output, labels)
            # update paramters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)
    
    print('train over')
    # 仅保存和加载模型参数(推荐使用)
    torch.save(net.state_dict(), 'params_sin.pkl')


def predict():
    result = list()
    net = GRUNet(input_size=1, hidden_size=64, out_dim=1)
    net.load_state_dict(torch.load('params_sin.pkl'))
    net.eval()
    train_x, train_y = GenetorNum.read_data_sin()
    x = train_x[-1:]
    x = Variable(torch.tensor(x).float())
    for i in range(100):
        out = net(x)
        out = out[0, -1, 0]
        result.append(out)
        x[0, :-1, 0] = x[0, 1:, 0]
        x[0, -1, 0] = out
    
    fig, ax = plt.subplots()
    ax.plot(range(len(result)), result, color='b')
    plt.show()


if __name__ == '__main__':
    train()
    predict()
