from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

W1 = tf.Variable(tf.zeros([540, 30]))
b1 = tf.Variable(tf.zeros([30]))

W2 = tf.Variable(tf.zeros([30, 15]))
b2 = tf.Variable(tf.zeros([15]))

x2 = tf.nn.softmax(tf.matmul(x, W1) + b1)
y = tf.nn.softmax(tf.matmul(x2, W2) + b2)

y_ = tf.placeholder(tf.float32, [None, 15])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#TODO: data zber zo subou po batchoch
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
