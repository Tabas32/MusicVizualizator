import tensorflow as tf
import time
import procesImages as input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def lrelu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

y_dim = 28 #549
z_dim = 100

#=================DISCRIMINATOR==========================
input_shape = input_data.WIDTH * input_data.HEIGHT
X = tf.placeholder(tf.float32, shape=[None, input_shape])
y = tf.placeholder(tf.float32, shape=[None, y_dim])

D_W1 = tf.Variable(xavier_init([input_shape + y_dim, 512]))
D_b1 = tf.Variable(tf.zeros(shape=[512]))

D_W2 = tf.Variable(xavier_init([512, 128]))
D_b2 = tf.Variable(tf.zeros(shape=[128]))

D_W3 = tf.Variable(xavier_init([128, 128]))
D_b3 = tf.Variable(tf.zeros(shape=[128]))

D_W4 = tf.Variable(xavier_init([128, 1]))
D_b4 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_W3, D_W4, D_b1, D_b2, D_b3, D_b4]


#================GENERATOR==============================
Z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim + y_dim, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))

G_W2 = tf.Variable(xavier_init([128, 128]))
G_b2 = tf.Variable(tf.zeros(shape=[128]))

G_W3 = tf.Variable(xavier_init([128, 512]))
G_b3 = tf.Variable(tf.zeros(shape=[512]))

G_W4 = tf.Variable(xavier_init([512, input_shape]))
G_b4 = tf.Variable(tf.zeros(shape=[input_shape]))

theta_G = [G_W1, G_W2, G_W3, G_W4, G_b1, G_b2, G_b3, G_b4]


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z, y):
    inputs = tf.concat(axis=1, values=[z, y])
    G_h1 = lrelu(tf.matmul(inputs, G_W1) + G_b1, 0.3)
    G_h2 = lrelu(tf.matmul(G_h1, G_W2) + G_b2, 0.3)
    G_h3 = lrelu(tf.matmul(G_h2, G_W3) + G_b3, 0.3)
    G_log_prob = tf.matmul(G_h3, G_W4) + G_b4
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x, y):
    inputs = tf.concat(axis=1, values=[x,y])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_h3 = tf.nn.relu(tf.matmul(D_h2, D_W3) + D_b3)
    D_prob = tf.matmul(D_h3, D_W4) + D_b4

    return D_prob


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(input_data.HEIGHT, input_data.WIDTH), cmap='Greys_r')

    return fig


G_sample = generator(Z,y)

D_real = discriminator(X,y)
D_fake = discriminator(G_sample, y)

D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
G_loss = -tf.reduce_mean(D_fake)

G_smr = tf.summary.scalar('G_loss', G_loss)
D_smr = tf.summary.scalar('D_loss', D_loss)

D_solver = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(-D_loss, var_list=theta_D)
G_solver = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(G_loss, var_list=theta_G)

clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

mb_size = 20 # batch size?

#  row of data [0] =  image
#              [1] =  song
data = np.load("..\\mini_data_S.npy")

with tf.Session() as sess:
    G_merge = tf.summary.merge([G_smr])
    D_merge = tf.summary.merge([D_smr])
    train_writer = tf.summary.FileWriter('\\tmp\\train\\11', sess.graph)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, '\\tmp\\w_cgan_model3.ckpt')

    if not os.path.exists('out/'):
        os.makedirs('out/')

    i = 0

    G_loss_curr = 0
    D_loss_curr = 0

    epochs = 10000
    for it in range(epochs+1):
        if it % (epochs/10) == 0:
            y_sample = input_data.next_batch(16, data)
            y_sample = np.array(list(y_sample[:, 1]), dtype=np.float)
            
            samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, z_dim), y:y_sample})
          
            fig = plot(samples)

            name = time.strftime('%d_%m_%Y_%H_%M_%S')
            name = "out/" + name + ".png"
            plt.savefig(name, bbox_inches='tight')
            i += 1
            plt.close(fig)

        batch = input_data.next_batch(mb_size, data)
        X_mb = np.array(list(batch[:, 0]), dtype=np.int)
        y_mb = np.array(list(batch[:, 1]), dtype=np.float)

        for _ in range(5):
            batch = input_data.next_batch(mb_size, data)
            X_mb = np.array(list(batch[:, 0]), dtype=np.int)
            y_mb = np.array(list(batch[:, 1]), dtype=np.float)

            D_summary, _, D_loss_curr, _ = sess.run(
                    [D_merge, D_solver, D_loss, clip_D], 
                    feed_dict={X: X_mb, Z: sample_Z(mb_size, z_dim), y:y_mb}
            )

        G_summary, _, G_loss_curr = sess.run(
                [G_merge, G_solver, G_loss], 
                feed_dict={Z: sample_Z(mb_size, z_dim), y:y_mb}
        )


        train_writer.add_summary(D_summary, it)
        train_writer.add_summary(G_summary, it)

        if it % (epochs/10) == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'.format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()

        if it % (epochs/10) == 0:
            save_path = saver.save(sess, '\\tmp\\w_cgan_model3.ckpt')
            print("Model saved in path: %s" % save_path)

#tensorboard --logdir=PATH
