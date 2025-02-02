import tensorflow as tf
import procesImages as input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


input_shape = input_data.WIDTH * input_data.HEIGHT
X = tf.placeholder(tf.float32, shape=[None, input_shape])

D_W1 = tf.Variable(xavier_init([input_shape, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_W2 = tf.Variable(xavier_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


Z = tf.placeholder(tf.float32, shape=[None, 100])

G_W1 = tf.Variable(xavier_init([100, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))

G_W2 = tf.Variable(xavier_init([128, input_shape]))
G_b2 = tf.Variable(tf.zeros(shape=[input_shape]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


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


G_sample = generator(Z)

D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# -------------------
#with tf.name_scope('D_loss'):
#    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
#    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
#    D_loss = D_loss_real + D_loss_fake

    #tf.summary.scalar('D_loss', D_loss)

#with tf.name_scope('G_loss'):
#    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)), name='G_loss')

G_smr = tf.summary.scalar('G_loss', G_loss)
D_smr = tf.summary.scalar('D_loss', D_loss)

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

mb_size = 20 # batch size?
Z_dim = 100

data = input_data.load_imgs_as_np()

with tf.Session() as sess:
    G_merge = tf.summary.merge([G_smr])
    D_merge = tf.summary.merge([D_smr])
    train_writer = tf.summary.FileWriter('\\tmp\\train\\3', sess.graph)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, '\\tmp\\model.ckpt')

    if not os.path.exists('out/'):
        os.makedirs('out/')

    i = 0

    for it in range(1000):
        if it % 100 == 0:
            samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
            
            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

        X_mb = input_data.next_batch(mb_size, data)

        D_summary, _, D_loss_curr = sess.run([D_merge, D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
        G_summary, _, G_loss_curr = sess.run([G_merge, G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

        train_writer.add_summary(D_summary, it)
        train_writer.add_summary(G_summary, it)

        if it % 100 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'.format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()

        if it % 100 == 0:
            save_path = saver.save(sess, '\\tmp\\model.ckpt')
            print("Model saved in path: %s" % save_path)

#tensorboard --logdir=PATH
