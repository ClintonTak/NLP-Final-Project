# title          : Vectorize_Pos_Tags.py
# description    : This function turns English POS tags from POS_tags_essay_responses.txt into vectors
# author         : Daniel Brett, Lauren Becker, Isaiah Rawlinson, & Clinton Tak
# date           : May 1, 2018
# usage          : Part of NLI Project
# python_version : 3.x
# ==================================================

vector_tags = open("../Data/TOEFL/POSvectorTags.txt", "w")
posFile = open("../Data/TOEFL/Pos_tags_essay_responses.txt", "r")
posList = [open('../Data/TOEFL/Pos_Tags.txt').read().split()]
pList = posList[0]

for line in posFile:
	addList=[]
	for item in line.split():
		if item in pList:
			addList.append(pList.index(item))
	vector_tags.write(str(addList))
	vector_tags.write("\n")