# title          : dnn.py
# description    : Convolutional Neural Network in TFLearn
# author         : Becker, Brett, Tak, and Rawlinson
# date           : Thursday, 26 April 2018.
# python version : 3.6.5
# ==================================================

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import read_data

"""
TF in this example expects (list, int) pairs so I changed the one-hot
vectors (to ints 0 - 5) when reading the data in
"""
trainX, trainY = read_data.training_data()
testX, testY = read_data.testing_data()

# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
# Converting labels to binary vectors
"""
As far as can tell, nb_classes is the number of possible values our
data can have. So in the imbd case it's either positive/negative but
we have classes for each language, so 6
"""
trainY = to_categorical(trainY, nb_classes=6)
testY = to_categorical(testY, nb_classes=6)

# Network building
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
# Same thing here as with nb_classes, need to change to 6
net = tflearn.fully_connected(net, 6, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY,
          validation_set=(testX, testY),
          show_metric=True,
          batch_size=32)
