import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf

X = np.arange(0, 5.0, 0.1)
with tf.Session() as sess:
    Y = tf.nn.softmax(X).eval()
plt.plot(X, Y)
plt.ylim(-0.05, 0.15)
plt.show()
