import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import procesImages as input_data
import time


mb_size = 20
z_dim = 100
X_dim = input_data.WIDTH * input_data.HEIGHT
y_dim = 28 #549
h_dim = 128
c = 0
lr = 1e-3


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
        plt.imshow(sample.reshape(input_data.HEIGHT, input_data.WIDTH), cmap='gray')

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# =============================== Q(z|X) ======================================

X = tf.placeholder(tf.float32, shape=[None, X_dim])
c = tf.placeholder(tf.float32, shape=[None, y_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

Q_W1 = tf.Variable(xavier_init([X_dim + y_dim, h_dim]))
Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

Q_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
Q_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

Q_W3 = tf.Variable(xavier_init([h_dim, h_dim]))
Q_b3 = tf.Variable(tf.zeros(shape=[h_dim]))

Q_W2_mu = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_mu = tf.Variable(tf.zeros(shape=[z_dim]))

Q_W2_sigma = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_sigma = tf.Variable(tf.zeros(shape=[z_dim]))


def Q(X, c):
    inputs = tf.concat(axis=1, values=[X, c])
    h1 = tf.nn.relu(tf.matmul(inputs, Q_W1) + Q_b1)
    h2 = tf.nn.relu(tf.matmul(h1, Q_W2) + Q_b2)
    h3 = tf.nn.relu(tf.matmul(h2, Q_W3) + Q_b3)
    z_mu = tf.matmul(h3, Q_W2_mu) + Q_b2_mu
    z_logvar = tf.matmul(h3, Q_W2_sigma) + Q_b2_sigma
    return z_mu, z_logvar


def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

P_W1 = tf.Variable(xavier_init([z_dim + y_dim, h_dim]))
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
P_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W3 = tf.Variable(xavier_init([h_dim, h_dim]))
P_b3 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W4 = tf.Variable(xavier_init([h_dim, h_dim]))
P_b4 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W5 = tf.Variable(xavier_init([h_dim, h_dim]))
P_b5 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W6 = tf.Variable(xavier_init([h_dim, X_dim]))
P_b6 = tf.Variable(tf.zeros(shape=[X_dim]))


def P(z, c):
    inputs = tf.concat(axis=1, values=[z, c])
    h1 = tf.nn.relu(tf.matmul(inputs, P_W1) + P_b1)
    h2 = tf.nn.relu(tf.matmul(h1, P_W2) + P_b2)
    h3 = tf.nn.relu(tf.matmul(h2, P_W3) + P_b3)
    h4 = tf.nn.relu(tf.matmul(h3, P_W4) + P_b4)
    h5 = tf.nn.relu(tf.matmul(h4, P_W5) + P_b5)
    logits = tf.matmul(h5, P_W6) + P_b6
    prob = tf.nn.sigmoid(logits)
    return prob, logits


# =============================== TRAINING ====================================

z_mu, z_logvar = Q(X, c)
z_sample = sample_z(z_mu, z_logvar)
_, logits = P(z_sample, c)

# Sampling from random z
X_samples, _ = P(z, c)

# E[log P(X|z)]
recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1)
# D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
# VAE loss
vae_loss = tf.reduce_mean(recon_loss + kl_loss)

solver = tf.train.AdamOptimizer().minimize(vae_loss)
summary = tf.summary.scalar('VAE_loss', vae_loss)

data = np.load("..\\data_R3.npy")

with tf.Session() as sess:
    merge = tf.summary.merge([summary])
    train_writer = tf.summary.FileWriter('\\tmp\\train_vae\\12', sess.graph)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, '\\tmp\\model\\cvae_model12.ckpt')

    if not os.path.exists('outV/'):
        os.makedirs('outV/')

    i = 0

    epochs = 100000
    for it in range(epochs):
        batch = input_data.next_batch(mb_size, data)
        X_mb = np.array(list(batch[:, 0]), dtype=np.int)
        y_mb = np.array(list(batch[:, 1]), dtype=np.float)

        summary, _, loss = sess.run([merge, solver, vae_loss], feed_dict={X: X_mb, c: y_mb})

        train_writer.add_summary(summary, it)
        if it % (epochs/10) == 0:
            print('Iter: {}'.format(it))
            print('Loss: {:.4}'. format(loss))
            print()

            save_path = saver.save(sess, '\\tmp\\model\\cvae_model12.ckpt')
            print("Model saved in path: %s" % save_path)

            #y = np.zeros(shape=[16, y_dim])
            y = np.random.rand(16, y_dim) - 0.5
            #y[:, np.random.randint(0, y_dim)] = 1.
            #y = input_data.next_batch(16, data)
            #y = np.array(list(y[:, 1]), dtype=np.float)

            samples = sess.run(X_samples,
                               feed_dict={z: np.random.randn(16, z_dim), c: y})

            fig = plot(samples)

            name = time.strftime('%d_%m_%Y_%H_%M_%S')
            name = "outV/" + name + ".png"
            plt.savefig(name, bbox_inches='tight')
            i += 1
            plt.close(fig)
