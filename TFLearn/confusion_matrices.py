# title          : confusion_matrices.py
# description    : Display confusion_matrices
# author         : Isaiah Rawlinson
# date           : Wednesday,  9 May 2018.
# python version : 3.6.5
# ==================================================

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
# class_names = iris.target_names


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()


def clean_file(filename):
    f = open(filename, 'r')
    data_txt = f.read()
    data_txt = data_txt.replace('[', '')
    data_txt = data_txt.replace(']', '')
    data_txt = data_txt.replace('\n ', '\n')
    f = open(filename, 'w')
    f.write(data_txt)
    f.close()


def parse_mats(directory, classes):
    model = ''
    for f in os.listdir(directory):
        filename = os.fsdecode(f)
        model = filename.replace('.txt', '')
        filename = os.path.join(directory, filename)
        clean_file(filename)

        cnf_matrix = np.loadtxt(filename, dtype=np.int16)
        plot_confusion_matrix(cnf_matrix,
                              classes=classes,
                              title=model,
                              normalize=True)


caes_classes = [
    "Arabic",
    "Chinese",
    "English",
    "French",
    "Portuguese",
    "Russian"
]
toefl_classes = [
    'ARA', 'DEU', 'FRA', 'HIN', 'ITA', 'JPN',
    'KOR', 'SPA', 'TEL', 'TUR', 'ZHO'
]

parse_mats('ConfusionMatrices/CAES', caes_classes)
# parse_mats('ConfusionMatrices/TOEFL', toefl_classes)
