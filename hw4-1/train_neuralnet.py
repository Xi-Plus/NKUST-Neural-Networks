# coding: utf-8
import sys
import os
sys.path.append(os.pardir)

import numpy as np
from keras.datasets import cifar10
from two_layer_net import TwoLayerNet

# データの読み込み
(x_train, t_train), (x_test, t_test) = cifar10.load_data()
print('train:', len(x_train))
print('test :', len(x_test))

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

from keras.utils import np_utils
t_train = np_utils.to_categorical(t_train)
t_test = np_utils.to_categorical(t_test)

network = TwoLayerNet(input_size=3072, hidden_size=100, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc, y1, t1 = network.accuracy(x_train, t_train)
        test_acc, y2, t2 = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('{} {:.4f} {:.4f}'.format(i, train_acc, test_acc))

        print('train:')
        ans = [[0 for i in range(10)] for j in range(10)]
        for _y, _t in zip(y1, t1):
            ans[_y][_t] += 1
        print('{:15}'.format('predict \\ real'), end='')
        for i in range(10):
            print('{:5d}'.format(i), end='')
        print()
        for i in range(10):
            print('{:15}'.format(i), end='')
            for v2 in ans[i]:
                print('{:5d}'.format(v2), end='')
            print()

        print('test:')
        ans = [[0 for i in range(10)] for j in range(10)]
        for _y, _t in zip(y2, t2):
            ans[_y][_t] += 1
        print('{:15}'.format('predict \\ real'), end='')
        for i in range(10):
            print('{:5d}'.format(i), end='')
        print()
        for i in range(10):
            print('{:15}'.format(i), end='')
            for v2 in ans[i]:
                print('{:5d}'.format(v2), end='')
            print()
