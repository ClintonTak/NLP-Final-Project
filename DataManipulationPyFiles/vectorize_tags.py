# title          : Vectorize_Tags.py
# description    : This function turns Spanish POS tags into vectors
# author         : Daniel Brett, Lauren Becker, Isaiah Rawlinson, & Clinton Tak
# date           : April 24th, 2018
# usage          : Part of NLI Project
# python_version : 3.x
# ==================================================


def vectorize_tags():
    vector_tags = open("../Data/vectorTags.txt", "w")
    posFile = open("../Data/tags.txt", "r").readlines()

    all_tags = []
    for line in posFile:
            for tag in line.split():
                    all_tags.append(tag)

    all_tags = list(set(all_tags))
    all_tags = {all_tags[i]: i for i in range(len(all_tags))}

    for line in posFile:
            line = [all_tags[tag] for tag in line.split()]
            vector_tags.write(str(line) + '\n')


vectorize_tags()
