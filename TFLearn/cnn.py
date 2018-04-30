# title          : cnn.py
# description    : Convolutional Neural Network in TFLearn
# author         : Becker, Brett, Tak, and Rawlinson
# date           : Thursday, 26 April 2018.
# python version : 3.6.5
# ==================================================

from __future__ import division, print_function, absolute_import
import sys
import clean_logs
import read_data

import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences

from datetime import datetime

# Set up command line arguments:
args = sys.argv[1:]
gpu_mode = '-gpu' in args
mac_os = '-mac' in args


startTime = datetime.now()

logfile = 'log.txt'

"""
TF in this example expects (list, int) pairs so I changed the one-hot
vectors (to ints 0 - 5) when reading the data in
"""
trainX, trainY = read_data.training_data()
testX, testY = read_data.testing_data()

# Data preprocessing
# Sequence padding
# length = max(len(max(trainX, key=len)), len(max(testX, key=len)))
length = 500
trainX = pad_sequences(trainX, maxlen=length, value=0.)
testX = pad_sequences(testX, maxlen=length, value=0.)
# Converting labels to binary vectors
"""
As far as can tell, nb_classes is the number of possible values our
data can have. So in the imbd case it's either positive/negative but
we have classes for each language, so 6
"""
trainY = to_categorical(trainY, nb_classes=6)
testY = to_categorical(testY, nb_classes=6)


# Network building
def build_network():
    net = tflearn.input_data([None, length])
    net = tflearn.embedding(net, input_dim=10000, output_dim=128)
    net = tflearn.lstm(net, 128, dropout=0.8)
    # Same thing here as with nb_classes, need to change to 6
    net = tflearn.fully_connected(net, 6, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')
    return net


# Training
def train(net):
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(trainX, trainY,
              validation_set=(testX, testY),
              show_metric=True,
              batch_size=32,
              snapshot_step=None)


if gpu_mode:
    tflearn.init_graph(num_cores=4, gpu_memory_fraction=0.5)
    with tf.device('/device:GPU:0'):
        net = build_network()
        # Redirect logs to a file
        if not mac_os:
            sys.stdout = open(logfile, 'w')
        train(net)
    sys.stdout = sys.__stdout__
else:
    net = build_network()
    if not mac_os:
        sys.stdout = open(logfile, 'w')
    train(net)
    sys.stdout = sys.__stdout__

if not mac_os:
    clean_logs.parse(logfile)

print('Process finished in', datetime.now() - startTime)
