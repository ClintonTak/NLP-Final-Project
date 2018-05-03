# title           : ngram_data.py
# description     : Make n-grams of data set
# author          : Becker, Brett, Tak, and Rawlinson
# date            : Wednesday,  2 May 2018.
# python_version  : 3.6.4
# ==================================================

from nltk import ngrams
from random import shuffle
import matplotlib.pyplot as plt


def make_ngrams(vector_file):
    tag_vectors = [eval(line) for line in open(vector_file).readlines()]

    all_grams = []
    n_grams = []
    for vector in tag_vectors:
        grams = ngrams(vector, 3, pad_right=False)
        # grams = [str(gram) for gram in grams]
        grams = list(sum(grams, ()))
        all_grams.extend(grams)
        n_grams.append(grams)

    return n_grams, max(all_grams) + 1

    '''
    # Convert individual n-grams to ints
    all_grams = list(set([j for i in n_grams for j in i]))
    all_grams = {all_grams[i]: i for i in range(len(all_grams))}

    n_grams = [[all_grams[g] for g in vector] for vector in n_grams]

    return n_grams, len(all_grams)
    '''


def save_ngrams_to_file(n_grams, file_name):
    ngram_file = open(file_name, 'w')

    for n_gram in n_grams:
        ngram_file.write(str(n_gram) + '\n')

    ngram_file.close()


filename = '../Data/vectorTags.txt'


n_grams, dims = make_ngrams(filename)


one_hot_file = open('../Data/EnglishLangOutVectors.txt')
one_hot_vectors = [eval(line) for line in one_hot_file.readlines()]

# Translate one-hot vectors to integers for tensorflow
states = [vector.index(1) for vector in one_hot_vectors]

# Connect the data so our tags and states don't get mixed up
for i in range(len(n_grams)):
    n_grams[i].append(states[i])

# Exclude longer, outlier essays
max_len = 1000
n_grams = [vector for vector in n_grams if (len(vector) < max_len+1 and
                                            len(vector) >= 25)]

# Shuffle data so we have more random samples
shuffle(n_grams)

# Split the data back up into tags and states
states = [vector.pop() for vector in n_grams]


# Split our data for training and testing
split = len(n_grams) // 10


def training_data():
    return n_grams[split:], states[split:]


def testing_data():
    return n_grams[:split], states[:split]


def length_distribution():
    lengths = [len(i) for i in n_grams]
    dic = {}
    for i in lengths:
        if i in dic:
            dic[i] += 1
        else:
            dic[i] = 1

    plt.bar(dic.keys(), dic.values(), 1.0)
    plt.show()
