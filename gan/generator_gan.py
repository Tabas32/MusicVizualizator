import tensorflow as tf
import time
import procesImages as input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
import librosa
import analyzer
import dataParser

input_song = ""
input_time = 0
if len(sys.argv) == 3:
    input_song = sys.argv[1]
    input_time = int(sys.argv[2])
    print("For analyzation: " + input_song + " at second " + str(input_time))
else:
    print("Invalid num of arguments")
    quit()


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

print('Scaning ' + input_song)
try:
    song, sr = librosa.load(input_song, offset = input_time, duration = 25, sr = 22050)
except:
    raise ValueError("Something wrong with load of " + input_song)

song_np = analyzer.analyzeLoadedSong(song, sr)
song_np = dataParser.normalizeSong("..\\mini_data_S_notNorm.npy", song_np)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, '\\tmp\\w_cgan_model3.ckpt')

    if not os.path.exists('out/'):
        os.makedirs('out/')

    samples = sess.run(G_sample, feed_dict={Z: sample_Z(1, z_dim), y:[song_np]})
  
    fig = plot(samples)

    name = time.strftime('%d_%m_%Y_%H_%M_%S')
    name = "out/" + name + ".png"
    plt.savefig(name, bbox_inches='tight')
    plt.close(fig)
