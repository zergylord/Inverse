import numpy as np
from utils import *
import tensorflow as tf
x_dim = 10
hid_dim = 100
z_dim = 3
n_i = 10
a_dim = 2
bandwidth = 0.1
def encode(inp,reuse=None):
    hid = tf.layers.dense(inp,hid_dim,activation=tf.nn.relu,name='hid1',reuse=reuse)
    z = tf.layers.dense(hid,z_dim,activation=None,name='hid2',reuse=reuse)
    return z
xPrime = tf.placeholder(tf.float32,shape=[None,x_dim])
x = tf.placeholder(tf.float32,shape=[None,x_dim])
xPrime_i = tf.placeholder(tf.float32,shape=[None,n_i,x_dim])
x_i = tf.placeholder(tf.float32,shape=[None,n_i,x_dim])
a = tf.placeholder(tf.float32,shape=[None,a_dim])
a_i = tf.placeholder(tf.float32,shape=[None,n_i,a_dim])

x_diff = encode(xPrime)-encode(x,True)
x_i_diff = encode(xPrime_i,True)-encode(x_i,True)
print(x_diff.shape,x_i_diff.shape)
#mb x 1 x n_1
print(x_diff.shape,x_i_diff.shape)
prob = rbf(tf.expand_dims(x_diff,1),x_i_diff,b=bandwidth)
a_hat = tf.squeeze(tf.matmul(prob,a_i),1)
loss = tf.losses.mean_squared_error(a,a_hat)

train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss) 


sess = tf.Session()
sess.run(tf.global_variables_initializer())

n_steps = int(1e6)
refresh = int(1e3)
mb_dim = 100

M = np.random.randn(x_dim,z_dim)
N = np.random.randn(a_dim,z_dim)
O = np.random.randn(z_dim,x_dim)
def forward(x,a):
    return np.matmul(np.matmul(x,M)+np.matmul(a,N),O)
cum_loss = 0
for i in range(n_steps):
    X_i = np.random.randn(mb_dim,n_i,x_dim)
    A_i = np.random.randn(mb_dim,n_i,a_dim)
    XPrime_i = forward(X_i,A_i)
    X = np.random.randn(mb_dim,x_dim)
    A = np.random.randn(mb_dim,a_dim)
    XPrime = forward(X,A)
    _,cur_loss = sess.run([train_op,loss],
            feed_dict={xPrime:XPrime,x:X,xPrime_i:XPrime_i,x_i:X_i,a:A,a_i:A_i})
    cum_loss += cur_loss
    if (i+1) % refresh == 0:
        print(i+1,cum_loss/refresh)
        cum_loss = 0



