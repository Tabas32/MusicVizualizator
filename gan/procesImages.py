import os
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

HEIGHT, WIDTH, CHANNEL = 64, 64, 1
BATCH_SIZE = 20

def process_img_S(image_name):
    img = Image.open(image_name)
    white_back = Image.new('RGB', img.size, (255,255,255))
    white_back.paste(img, mask = img.split()[3])

    gray = white_back.convert('1')
    small = gray.resize((HEIGHT, WIDTH))
    arr = np.array(small)
    flat_arr = arr.ravel()
    return (1*flat_arr)

def load_imgs_as_np():
    directory = "..\images_S"
    images = []
    
    for each in os.listdir(directory):
        img = os.path.join(directory, each)
        img_arr = process_img_S(img)
        images.append(img_arr)

    return np.asarray(images)

def next_batch(num, data):
    data_shuffle = np.copy(data)
    np.random.shuffle(data_shuffle)
    
    return data_shuffle[:num]

def process_img():
    img = Image.open('temp.jpg')

    gray = img.convert('1')
    arr = np.array(gray)
    return (1*arr)

def process_data():   
    current_dir = os.getcwd()
    # parent = os.path.dirname(current_dir)
    pokemon_dir = os.path.join(current_dir, 'data')
    images = []
    for each in os.listdir(pokemon_dir):
        images.append(os.path.join(pokemon_dir,each))
    # print images    
    all_images = tf.convert_to_tensor(images, dtype = tf.string)
    
    images_queue = tf.train.slice_input_producer(
                                        [all_images])
                                        
    content = tf.read_file(images_queue[0])
    image = tf.image.decode_jpeg(content, channels = CHANNEL)
    # sess1 = tf.Session()
    # print sess1.run(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta = 0.1)
    image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)
    # noise = tf.Variable(tf.truncated_normal(shape = [HEIGHT,WIDTH,CHANNEL], dtype = tf.float32, stddev = 1e-3, name = 'noise')) 
    # print image.get_shape()
    size = [HEIGHT, WIDTH]
    image = tf.image.resize_images(image, size)
    image.set_shape([HEIGHT,WIDTH,CHANNEL])
    # image = image + noise
    # image = tf.transpose(image, perm=[2, 0, 1])
    # print image.get_shape()
    
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    
    iamges_batch = tf.train.shuffle_batch(
                                    [image], batch_size = BATCH_SIZE,
                                    num_threads = 4, capacity = 200 + 3* BATCH_SIZE,
                                    min_after_dequeue = 200)
    num_images = len(images)

    return iamges_batch, num_images
