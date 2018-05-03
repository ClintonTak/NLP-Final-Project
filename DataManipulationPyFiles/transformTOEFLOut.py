# title          : transformEnglishLangOut.py
# description    : Transfrom output to one-hot vectors
# author         : Isaiah Rawlinson
# date           : Tuesday, 24 April 2018.
# python_version : 3.6.5
# ==================================================


english_out_file = '../Data/TOEFL/language_of_respondents.txt'
transformed_file = '../Data/TOEFL/transformed_language_of_respondents.txt'

with open(english_out_file) as f:
    content = f.readlines()
    content = [line.rstrip() for line in content]

unique_langs = sorted([x for x in set(content)])

langs = {}
for i in range(len(unique_langs)):
    langs[unique_langs[i]] = i


vectors = []
for line in content:
    vectors.append(langs[line])


f = open(transformed_file, 'w')
f.write(str(vectors))
f.close()
