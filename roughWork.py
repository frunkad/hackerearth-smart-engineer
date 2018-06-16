import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn

training_data = pd.read_csv("train_data.csv")
train_label = training_data['E']
del training_data['E']
train_data = training_data
pd.plotting.scatter_matrix(train_data,c=train_label,figsize=(15,15),hist_kwds={"bins":20},s=60,alpha=.8)


from sklearn.model_selection import train_test_split
# trainX, testX, trainy, testy = train_test_split(train_data.values,train_label.values)
trainX,trainy = train_data.values,train_label.values
# trainX = trainX.T.astype(np.float32)

def f(x):
    return - x[0]*0.5005045 - x[1]*0.18817 + x[2]*21.7971 - x[3]*0.64242 + 28.097
def p(x):
    return - x[0]* 0.59526034 - x[1]*0.16321517 + x[2]*2.64657414 -x[3]*1.02715982 + 1824.0367186430083
for i in range(100):
    print("{} \t {} \t {}".format(trainy[i],f(trainX[i]),p(trainX[i])))

W = tf.Variable(tf.random_uniform([1, 4], -10.0, 10.0))
b = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

h = tf.matmul(W,trainX) + b
cost  = tf.reduce_mean(tf.square(h - trainy))


a = tf.Variable(0.00321)
optimizer = tf.train.RMSPropOptimizer(a)
train = optimizer.minimize(cost)
init = tf.global_variables_initializer()
initl = tf.local_variables_initializer()


writer = tf.summary.FileWriter("./tensorboards-logs", graph=tf.get_default_graph())

with tf.Session() as sess:
    sess.run(initl)
    sess.run(init)
    for i in range(20001):
        sess.run(train)
        if i % 200 == 0:
            print(i,sess.run(cost),sess.run(cost),sess.run(W),sess.run(b))
    # sess.run(l)
    accuracy, accuracy_op = tf.metrics.accuracy(labels=tf.argmax(trainX, 0), predictions=tf.argmax(h, 0))
    sess.run([init,initl])
    print(sess.run(accuracy_op))


(array([-0.50283045, -0.1916399 , 21.977737  , -0.6420597 ], dtype=float32),
 10.8601885)