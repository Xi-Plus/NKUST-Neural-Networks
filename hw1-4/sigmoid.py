import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf

X = np.arange(-5.0, 5.0, 0.1)
with tf.Session() as sess:
    Y = tf.math.sigmoid(X).eval()
plt.plot(X, Y)
plt.ylim(-1.1, 1.1)
plt.show()
