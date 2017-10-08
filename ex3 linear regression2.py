import tensorflow as tf
x_data=[1,2,3]
y_data=[1,2,3]
W=tf.Variable(tf.random_uniform([1], -1,1))
b=tf.Variable(tf.random_uniform([1], -1, 1))
X=tf.placeholder(tf.float32, name="X")
Y=tf.placeholder(tf.float32, name="Y")

hypothesis0=W*X+b
hypothesis1=W*hypothesis0+b
hypothesis=W*hypothesis1+b

cost = tf.reduce_mean(tf.square(hypothesis-Y))
optimizer = tf.train.AdamOptimizer(learning_rate=1)
train_op=optimizer.minimize(cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1000):
        _, cost_val=sess.run([train_op, cost], feed_dict={X:x_data, Y: y_data})
        print(step, cost_val, sess.run(W), sess.run(b))
    print("X:5,Y=", sess.run(hypothesis, feed_dict={X:5}))
    print("X:5,Y=", sess.run(hypothesis, feed_dict={X:2.5}))
