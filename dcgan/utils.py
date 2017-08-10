from tensorflow.contrib.layers import xavier_initializer_conv2d
import matplotlib.image as mpimage
import matplotlib.pyplot as plt
from termcolor import colored
import tensorflow as tf
import numpy as np
import os


def lrelu(x, leak):
    return tf.maximum(x, leak * x)


def conv2d(input, filter_shape, strides=2, padding='SAME', name='conv'):
    with tf.variable_scope(name):
        w = tf.get_variable(name='w',
                            shape=filter_shape,
                            dtype=tf.float32,
                            initializer=xavier_initializer_conv2d())
        b = tf.get_variable(name='b',
                            shape=[filter_shape[-1]],
                            dtype=tf.float32,
                            initializer=tf.zeros_initializer())
        conv = tf.nn.conv2d(input=input,
                            filter=w,
                            strides=[1, strides, strides, 1],
                            padding=padding,
                            use_cudnn_on_gpu=True)
        return lrelu(conv + b, 0.02)


def batch_norm(input):
    return batch_norm(input)


def deconv2d(input, filter_shape, output_shape, stride, padding="SAME", name='deconv', use_tanh=False):
    with tf.variable_scope(name):
        w = tf.get_variable(name='w',
                            shape=filter_shape,
                            initializer=xavier_initializer_conv2d(),
                            dtype=tf.float32)
        b = tf.get_variable(name='b',
                            shape=[filter_shape[-2]],
                            initializer=tf.zeros_initializer(),
                            dtype=tf.float32)
        deconv = tf.nn.conv2d_transpose(value=input,
                                        filter=w,
                                        output_shape=output_shape,
                                        strides=[1, stride, stride, 1],
                                        padding=padding, name=name)
        if use_tanh:
            output = tf.nn.tanh(deconv + b)
        else:
            output = lrelu(deconv + b, 0.02)
        return output


def multiply(x, shape, name='multiply', activation='lrelu'):
    with tf.variable_scope(name):
        w = tf.get_variable(name='w',
                            dtype=tf.float32,
                            initializer=xavier_initializer_conv2d(),
                            shape=shape)
        b = tf.get_variable(name='b',
                            dtype=tf.float32,
                            initializer=tf.zeros_initializer(),
                            shape=[shape[-1]])
        output = tf.nn.xw_plus_b(x, w, b)
        if activation == 'sigmoid':
            output = tf.nn.sigmoid(output)
        elif activation == 'lrelu':
            output = lrelu(output, 0.02)

        return output


def sample_noise(shape):
    return tf.random_uniform(shape=shape, minval=-1, maxval=1)


def add_noise(input, shape):
    return input + sample_noise(shape)


def read_image(address, format):
    print('reading from : ', colored(address, 'blue'))
    image_dir = [address+name for name in os.listdir(address) if format in name.lower()]
    print(image_dir)
    images = [plt.imread(name) for name in image_dir]
    print(colored('reading finished.', 'green'))
    return np.array(images)
