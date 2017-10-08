import tensorflow as tf
X=tf.placeholder(tf.float32, [None,3])
print(X)

X_data=[[1,2,3],[4,5,6]]

W = tf.Variable(tf.random_normal([3,2]))
b = tf.Variable(tf.random_normal([2,1]))
W=tf.Variable([[0.1,0.1],[0.2,0.2],[0.3,0.3]])
expr = tf.matmul(X,W)+b

sess=tf.Session()
sess.run(tf.global_variables_initializer())

print("=== x_data===")
print(X_data)
print("===W===")
print(sess.run(W))
print("===b===")
print(sess.run(b))
print("===expr===")
print(sess.run(expr, feed_dict={X:X_data}))
sess.close()
