import tensorflow as tf
import numpy as np
x_data=np.array(
    [[1],[2.3],[3.5],[2.4],[1.6],[7.3],[-3.4],[-4.6]])
v_data=np.array(
    [[2.3],[1.5],[-2.3],[2.3],[-1],[1],[-3.2],[2.3]])
des_data=np.array(
    [[-1.3],[4.5],[-3.2],[1.3],[-1.5],[-1],[-1.2],[5.3]])

Destination=tf.placeholder(tf.float32)
Vel=tf.placeholder(tf.float32)#로봇의 속도'''
Rx=tf.placeholder(tf.float32)#갱신된 위치'''
dtime = tf.constant([[0.01]])

W1=tf.Variable(tf.random_uniform([3, 5], -0.01 , 0.01))
W2=tf.Variable(tf.random_uniform([5, 1], -0.01 , 0.01))

b1=tf.Variable(tf.zeros([5]))
b2=tf.Variable(tf.zeros([1]))

for step in range(10):
    
    VR=tf.concat([Vel, Rx, Destination],0)

    A=tf.add(tf.matmul(VR, W1), b1)
    A=tf.nn.relu(A)
    A=tf.add(tf.matmul(A, W2), b2)
    A=tf.nn.tanh(A)#가속도

    Vel = tf.add(tf.matmul(A, dtime) , Vel)
    Rx = tf.add(tf.matmul(A, dtime), Rx)
    
cost=tf.reduce_mean(tf.square(Destination-Rx))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op=optimizer.minimize(cost)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

for step in range(100):
    sess.run(train_op, feed_dict={Vel:v_data, Rx:x_data, Destination:des_data})
    if(step+1)%1==0:
        print(step+1, sess.run(cost, feed_dict={Vel:v_data, Rx:x_data, Destination:des_data}))
