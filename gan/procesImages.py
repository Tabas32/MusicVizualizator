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

def process_img_R(image_name):
    img = Image.open(image_name)
    white_back = Image.new('RGB', img.size, (255,255,255))
    white_back.paste(img, mask = img.split()[3])

    #gray = white_back.convert('L')
    small = white_back.resize((HEIGHT, WIDTH))
    arr = np.array(small)
    flat_arr = arr.ravel()
    return flat_arr/255

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

def arr_to_img(arr):
    im = Image.fromarray((arr).reshape(WIDTH, HEIGHT))
    return im.convert('RGB')
