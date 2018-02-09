import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

df = pd.read_csv('fer2013.csv')

labels = list(df['emotion'])
# images with shape (48, 48)
X = []
# labels
y = []

for i in range(df.shape[0]):
    img_pixels = df['pixels'][i]
    label = df['emotion'][i]

    img_pixels = img_pixels.split(' ')
    img_pixels = np.array(img_pixels, np.uint8)
    img = np.reshape(img_pixels, (48, 48))

    X.append(img)
    y.append(label)

X = np.array(X)
Xn = np.asarray(X, np.float32)
y = np.array(y)
x = tf.convert_to_tensor(X, np.float32)
with tf.Session() as sess:
    print(sess.run([x]))
    print(tf.shape(x))
