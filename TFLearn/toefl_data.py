# title          : toefl_data.py
# description    : Read in vectors from text files
# author         : Becker, Brett, Tak, and Rawlinson
# date           : Thursday,  3 May 2018.
# python version : 3.6.5
# ==================================================
from random import shuffle
import matplotlib.pyplot as plt

# Load the data from our vector files
tag_file = open('../Data/TOEFL/POSvectorTags.txt')
lang_file = open('../Data/TOEFL/transformed_language_of_respondents.txt')


# Convert to lists we can use
tag_vectors = [eval(line) for line in tag_file.readlines()]
states = eval(lang_file.readlines()[0])


# Connect the data so our tags and states don't get mixed up
for i in range(len(tag_vectors)):
    tag_vectors[i].append(states[i])


max_len = 600
# Exclude longer, outlier essays
tag_vectors = [v for v in tag_vectors if (len(v) < max_len+1 and
                                          len(v) >= 100)]

all_tags = []
for vector in tag_vectors:
    for tag in vector:
        all_tags.append(tag)

dims = max(list(set(all_tags))) + 1


# Shuffle data so we have more random samples
shuffle(tag_vectors)

# Split the data back up into tags and states
states = [vector.pop() for vector in tag_vectors]


# Split our data for training and testing
split = len(tag_vectors) // 10


def training_data():
    return tag_vectors[split:], states[split:]


def testing_data():
    return tag_vectors[:split], states[:split]


def length_distribution():
    lengths = [len(i) for i in tag_vectors]
    dic = {}
    for i in lengths:
        if i in dic:
            dic[i] += 1
        else:
            dic[i] = 1

    plt.bar(dic.keys(), dic.values(), 1.0)
    plt.show()


# length_distribution()
