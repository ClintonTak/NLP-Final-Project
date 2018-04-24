# title          : Vectorize_Tags.py
# description    : This function turns Spanish POS tags into vectors
# author         : Daniel Brett, Lauren Becker, Isaiah Rawlinson, & Clinton Tak
# date           : April 24th, 2018
# usage          : Part of NLI Project
# python_version : 3.x
# ==================================================

import numpy as np
import tensorflow as tf
from random import shuffle
from sklearn.feature_extraction.text import CountVectorizer

class vectorize_tags(object):
	def __init__(self, num):
		self.num=num

	def vectorize(self):
		vector_tags = open("vectorTags.txt", "w")
		posFile = open("tags.txt", "r")
		caesList = [open('CaesTags.txt').read().split()]
		cList = caesList[0]
		#POS = open('tags.txt').read().split()
		for line in posFile:
			addList=[]
			for item in line:
				if item in cList:
					addList.append(cList.index(item))
			vector_tags.write(str(addList))
			vector_tags.write("\n")
			
				


someTags = vectorize_tags(1)
someTags.vectorize()