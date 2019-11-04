import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf

X = np.arange(-5.0, 5.0, 0.1)
with tf.Session() as sess:
    Y = tf.nn.relu(X).eval()
plt.plot(X, Y)
plt.ylim(-1.1, 5.1)
plt.show()
