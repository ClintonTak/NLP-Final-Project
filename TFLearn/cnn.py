# title          : cnn.py
# description    : Convolutional Neural Network in TFLearn
# author         : Becker, Brett, Tak, and Rawlinson
# date           : Tuesday,  1 May 2018.
# python version : 3.6.5
# ==================================================

from __future__ import division, print_function, absolute_import
import sys
import clean_logs
import read_data

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences

from datetime import datetime
startTime = datetime.now()

"""
Set up command line arguments:
    -gpu        - use tensorflow-gpu
    -mac        - print results of training to console
    -epochs=x   - specify the number of epochs (e.g. -epochs=10)
    -op=n       - specify the optimizer (e.g. -op=rmsprop)
"""
args = sys.argv[1:]
gpu_mode = '-gpu' in args
mac_os = '-mac' in args
epochs = 5
optimizer = 'adam'
for arg in args:
    if '-epochs=' in arg:
        epochs = int(arg[8:])
    if '-op=' in arg:
        optimizer = arg[4:]


file_header = 'cnn_' + optimizer
# Specify log file
logfile = 'Logs/' + file_header + '.txt'


trainX, trainY = read_data.training_data()
testX, testY = read_data.testing_data()

# Data preprocessing
# Sequence padding
length = 500
trainX = pad_sequences(trainX, maxlen=length, value=0.)
testX = pad_sequences(testX, maxlen=length, value=0.)
# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=6)
testY = to_categorical(testY, nb_classes=6)


# Building convolutional network
def build_network(optimizer):
    net = input_data(shape=[None, length], name='input')
    net = tflearn.embedding(net, input_dim=10000, output_dim=128)
    branch1 = conv_1d(net, 128, 3,
                      padding='valid',
                      activation='relu',
                      regularizer="L2")
    branch2 = conv_1d(net, 128, 4,
                      padding='valid',
                      activation='relu',
                      regularizer="L2")
    branch3 = conv_1d(net, 128, 5,
                      padding='valid',
                      activation='relu',
                      regularizer="L2")
    net = merge([branch1, branch2, branch3], mode='concat', axis=1)
    net = tf.expand_dims(net, 2)
    net = global_max_pool(net)
    net = dropout(net, 0.5)
    net = fully_connected(net, 6, activation='softmax')
    net = regression(net,
                     optimizer=optimizer,
                     learning_rate=0.001,
                     loss='categorical_crossentropy',
                     name='target')

    return net


# Training
def train(net):
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(trainX, trainY,
              n_epoch=epochs,
              shuffle=True,
              validation_set=(testX, testY),
              show_metric=True,
              batch_size=32)

    model.save('Models/' + file_header + '/' + file_header + '.tfl')


if gpu_mode:
    tflearn.init_graph(num_cores=4, gpu_memory_fraction=0.5)
    with tf.device('/device:GPU:0'):
        net = build_network(optimizer)
        # Redirect logs to a file
        if not mac_os:
            sys.stdout = open(logfile, 'w')
        train(net)
    sys.stdout = sys.__stdout__
else:
    net = build_network(optimizer)
    if not mac_os:
        sys.stdout = open(logfile, 'w')
    train(net)
    sys.stdout = sys.__stdout__

if not mac_os:
    clean_logs.parse(logfile)


print('Process finished in', datetime.now() - startTime)
