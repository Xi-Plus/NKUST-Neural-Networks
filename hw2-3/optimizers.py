# coding: utf-8
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf


def optimizers(lr=0.01, step_num=100):
    x_history = []

    for i in range(step_num):
        x_history.append([varx0.eval(sess), varx1.eval(sess)])

        sess.run(train_step)

    return np.array(x_history)


names = ['SGD', 'Momentum', 'AdaGrad', 'RMSProp', 'Adam']
for i in range(5):
    varx0 = tf.Variable(-3.0)
    varx1 = tf.Variable(4.0)
    output = tf.add(tf.square(varx0), tf.square(varx1))

    if i == 0:
        train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(output)
    elif i == 1:
        train_step = tf.train.MomentumOptimizer(learning_rate=0.03, momentum=0.9).minimize(output)
    elif i == 2:
        train_step = tf.train.AdagradOptimizer(learning_rate=1).minimize(output)
    elif i == 3:
        train_step = tf.train.RMSPropOptimizer(learning_rate=0.2).minimize(output)
    elif i == 4:
        train_step = tf.train.AdamOptimizer(learning_rate=0.8).minimize(output)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    lr = 0.1
    step_num = 20
    x_history = optimizers(lr=lr, step_num=step_num)

    plt.figure(names[i])

    plt.plot([-5, 5], [0, 0], '--b')
    plt.plot([0, 0], [-5, 5], '--b')
    plt.plot(x_history[:, 0], x_history[:, 1], 'o')

    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel("X0")
    plt.ylabel("X1")
plt.show()
