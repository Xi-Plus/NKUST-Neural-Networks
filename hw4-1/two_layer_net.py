# coding: utf-8
import sys
import os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict


class MyNet:
    layer_size = None

    def __init__(self, layer_size, weight_init_std=0.01):
        self.layer_size = layer_size
        # 重みの初期化
        self.params = {}
        for i in range(1, len(layer_size)):
            self.params['W{}'.format(i)] = weight_init_std * np.random.randn(layer_size[i - 1], layer_size[i])
            self.params['b{}'.format(i)] = np.zeros(layer_size[i])

        # レイヤの生成
        self.layers = OrderedDict()
        for i in range(1, len(layer_size)):
            self.layers['Affine{}'.format(i)] = Affine(self.params['W{}'.format(i)], self.params['b{}'.format(i)])
            if i < len(layer_size) - 1:
                self.layers['Relu{}'.format(i)] = Relu()

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy, y, t

    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        def loss_W(W): return self.loss(x, t)

        grads = {}
        for i in range(1, len(self.layer_size)):
            grads['W{}'.format(i)] = numerical_gradient(loss_W, self.params['W{}'.format(i)])
            grads['b{}'.format(i)] = numerical_gradient(loss_W, self.params['b{}'.format(i)])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for i in range(1, len(self.layer_size)):
            grads['W{}'.format(i)], grads['b{}'.format(i)] = self.layers['Affine{}'.format(i)].dW, self.layers['Affine{}'.format(i)].db

        return grads
