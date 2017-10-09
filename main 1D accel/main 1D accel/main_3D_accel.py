import tensorflow as tf
import numpy as np

r_data=np.array(
    [[-1.3, 0.5, 2.3],[4.5, 4.2, -4.5],[-3.2, 0.2, -3.4],[1.3, 9.1, 4.6],[-1.5, 0.3, 0.3],[-1, -4.3, 2.3],[-1.2, 3.2, -3.5],[5.3, 4.8, -3]])
v_data=np.array(
    [[1, -1, -2],[2.3, 3.2, 4.3],[3.5, -3.2, -1],[2.4, 0.3, -3.2],[1.6, 4.3, 3.4],[7.3, -3.2, -5.1],[-3.4, 2.5, -3.2],[-4.6, 3.3, -1.5]])
des_data=np.array(
    [[2.3, 1.4, 3.2],[1.5,-3.2,2.3 ],[-2.3, 2.3, 0.1],[2.3, 3.3, 2.0],[-1, -2.3, -2.1],[1, 3.2, -5.2],[-3.2, 2.2, -5.3],[2.3, -4.3, -4.3]])

global_step = tf.Variable(0, trainable=False, name='global_step')

Destination=tf.placeholder(tf.float32, [None, 3])
Vel=tf.placeholder(tf.float32, [None, 3])#로봇의 속도'''
Rx=tf.placeholder(tf.float32, [None,3])#갱신된 위치'''

dtime = tf.constant(0.1)

print('r_data.shape = ',r_data.shape)
print('v_data.shape = ',v_data.shape)
print('des_data.shape = ',des_data.shape)

W1=tf.Variable(tf.random_normal([9, 20], -0.01 , 0.01))
W2=tf.Variable(tf.random_normal([20, 3], -0.01 , 0.01))

b1=tf.Variable(tf.zeros([10]))
b2=tf.Variable(tf.zeros([3]))

V_now=Vel
Rx_now=Rx

for _ in range(100):
    L0=tf.concat([V_now, Rx_now, Destination],axis=1)

    L1=tf.add(tf.matmul(L0, W1), b1)
    L1=tf.nn.relu(L1)

    L2=tf.add(tf.matmul(L1, W2), b2)

    Accel=L2 #가속도 갱신
    
    V_now = tf.add(tf.scalar_mul(dtime , Accel) , V_now) #속도 갱신

    Rx_now = tf.add(tf.scalar_mul(dtime , V_now), Rx_now) #위치 갱신

cost=tf.reduce_mean(tf.square(Rx_now-Destination))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost, global_step=global_step)

init=tf.global_variables_initializer()
sess=tf.Session()
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('./accel')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

for step in range(100):
    sess.run(train_op, feed_dict={Vel:v_data, Rx:r_data, Destination:des_data})
    if(step+1)%10==0:
        print('step: %d'%sess.run(global_step), 
        'cost: %.3f'%sess.run(cost, feed_dict={Vel:v_data, Rx:r_data, Destination:des_data}))
'''x_test=np.array(
    [[5],[1.3],[-3.2],[3.2],[-2.6],[5.0],[-1.5],[-0.4]])
v_test=np.array(
    [[0.1],[3.1],[2.3],[4.2],[-1.5],[1.7],[-1.2],[-2.3]])
des_test=np.array(
    [[-1.2],[4.2],[3.2],[-1.3],[1.6],[-9],[-3.2],[-5.3]])

print('test cost= ', sess.run(cost,  feed_dict={Vel:v_test, Rx:x_test, Destination:des_test}))'''

saver.save(sess, './accel/dnn.ckpt', global_step=global_step)

