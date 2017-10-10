import tensorflow as tf
x_data=[0,10,20,30,34,36,38 ,40 ,42 ,44,46,50  ,60  ,70  ,80  ,84  ,86  ,88  ,90  ,92  ,94  ,96  ,100 ,110 ,120 ,130 ,140 ,150 ,160 ,170 ,180]
y_data=[1600,1384,912 ,370 ,196 ,138 ,72  ,39  ,11  ,2.1 ,5.6 ,45,380 ,965 ,1447,1586,1625,1644,1648,1619,1585,1550,1410,925 ,371,42  ,59  ,425 ,958 ,1385,1433]
W=tf.Variable(tf.random_uniform([1], 1610,1611))
b=tf.Variable(tf.random_uniform([1], -0.1, 0.1))
X=tf.placeholder(tf.float32, name="X")
Y=tf.placeholder(tf.float32, name="Y")
hypothesis=W*tf.cos(X*3.1415926/360.)*tf.cos(X*3.141592/360.)
cost = tf.reduce_mean(tf.square(hypothesis-Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_op=optimizer.minimize(cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1000):
        _, cost_val=sess.run([train_op, cost], feed_dict={X:x_data, Y: y_data})
        print(step, cost_val, sess.run(W), sess.run(b))
    print("X:5,Y=", sess.run(hypothesis, feed_dict={X:5}))
    print("X:2.5,Y=", sess.run(hypothesis, feed_dict={X:2.5}))
