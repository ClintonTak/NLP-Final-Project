# title          : read_data.py
# description    : Read in vectors from text files
# author         : Isaiah Rawlinson
# date           : Thursday, 26 April 2018.
# python version : 3.6.5
# ==================================================


one_hot_file = open('../Data/EnglishLangOutVectors.txt')
tag_file = open('../Data/vectorTags.txt')

tag_vectors = [eval(line) for line in tag_file.readlines()]
one_hot_vectors = [eval(line) for line in one_hot_file.readlines()]

split = len(one_hot_vectors) // 2
print(split)


def training_data():
    return tag_vectors[:split], one_hot_vectors[:split]


def testing_data():
    return tag_vectors[split:], one_hot_vectors[split:]
