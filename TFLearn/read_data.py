# title          : read_data.py
# description    : Read in vectors from text files
# author         : Becker, Brett, Tak, and Rawlinson
# date           : Thursday, 26 April 2018.
# python version : 3.6.5
# ==================================================


tag_file = open('../Data/vectorTags.txt')
one_hot_file = open('../Data/EnglishLangOutVectors.txt')

tag_vectors = [eval(line) for line in tag_file.readlines()]
one_hot_vectors = [eval(line) for line in one_hot_file.readlines()]
states = [vector.index(1) for vector in one_hot_vectors]

split = len(one_hot_vectors) // 2
print(split)


def training_data():
    return tag_vectors[:split], states[:split]


def testing_data():
    return tag_vectors[split:], states[split:]
