from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

def next_batch(num, data):
    data_shuffle = np.copy(data)
    np.random.shuffle(data_shuffle)
    
    return data_shuffle[:num]

def split_data(num, data):
    data_shuffle = np.copy(data)
    np.random.shuffle(data_shuffle)
    
    return data_shuffle[:num], data_shuffle[num:]

data = np.load("..\\data_CC.npy")
train_data, test_data = split_data(80, data)

x = tf.placeholder(tf.float32, [None, 549])

W1 = tf.Variable(tf.zeros([549, 15]))
b1 = tf.Variable(tf.zeros([15]))

#W2 = tf.Variable(tf.zeros([30, 15]))
#b2 = tf.Variable(tf.zeros([15]))

y = tf.nn.softmax(tf.matmul(x, W1) + b1)
#y = tf.nn.softmax(tf.matmul(h1, W2) + b2)

y_ = tf.placeholder(tf.float32, [None, 15])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
error = tf.summary.scalar('Loss', cross_entropy)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

merge = tf.summary.merge([error])
train_writer = tf.summary.FileWriter('\\tmp\\cc_train\\2', sess.graph)

saver = tf.train.Saver()
saver.restore(sess, '\\tmp\\cc\\cc_model1.ckpt')

mb_size = 20
epochs = 100000
for epoch in range(epochs + 1):
    batch = next_batch(mb_size, train_data)
    X_mb = np.array(list(batch[:, 1]), dtype=np.float)
    y_mb = np.array(list(batch[:, 0]), dtype=np.float)

    summary, _, loss = sess.run([merge, train_step, cross_entropy], feed_dict={x: X_mb, y_: y_mb})
    train_writer.add_summary(summary, epoch)

    if epoch % (epochs / 10) == 0:
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        X_test = np.array(list(test_data[:, 1]), dtype=np.float)
        y_test = np.array(list(test_data[:, 0]), dtype=np.float)
        print("Epoch: " + str(epoch))
        print(20*"-")
        print("Current loss: " + str(loss))
        print("Accuracy: " + str(sess.run(accuracy, feed_dict={x: X_test, y_: y_test})))

        save_path = saver.save(sess, '\\tmp\\cc\\cc_model1.ckpt')
        print("Model saved in path: %s" % save_path)
        print()
