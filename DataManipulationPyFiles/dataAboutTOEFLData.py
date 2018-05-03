# -*- coding: utf-8 -*-
# @Author: Clinton Tak, Lauren Becker, Isaiah Rawlinson, Daniel Brett
# @Date:   2018-05-01 14:00:35
# @Last Modified by:   Clinton Tak
# @Last Modified time: 2018-05-02 21:36:21
import re 
import numpy as np 

import numpy as np
import scipy.stats as stats
import pylab as pl

#opening data 
languagesFiles = open('../Data/TOEFL/language_of_respondents.txt', 'r')
toeflPOSTagFile = open('../Data/TOEFL/POS_tags_essay_responses.txt', 'r')
uniqueTagList = open('../Data/TOEFL/Pos_Tags.txt')
intPOSTagFile = open('../Data/TOEFL/POSvectorTags.txt')

Languages = languagesFiles.readlines()
POSTags = toeflPOSTagFile.readlines()
uniqueTags = uniqueTagList.readlines()
intPOSTags = intPOSTagFile.readlines()
##getting average number of POS tags by language 
germanPOSCount = []
turkishPOSCount = []
frenchPOSCount = []
arabicPOSCount = []
koreanPOSCount = []
chinesePOSCount = []
hindiPOSCount = []
spanishPOSCount = []
italianPOSCount = []
japanesePOSCount = []
teluguPOSCount = []


for j in range(0, len(Languages)):
	if "DEU" in Languages[j]:
		germanPOSCount.append(len(eval(intPOSTags[j])))
	if "TUR" in Languages[j]:
		turkishPOSCount.append(len(eval(intPOSTags[j])))
	if "FRA" in Languages[j]:
		frenchPOSCount.append(len(eval(intPOSTags[j])))
	if "ARA" in Languages[j]:
		arabicPOSCount.append(len(eval(intPOSTags[j])))
	if "KOR" in Languages[j]:
		koreanPOSCount.append(len(eval(intPOSTags[j])))
	if "ZHO" in Languages[j]:
		chinesePOSCount.append(len(eval(intPOSTags[j])))
	if "HIN" in Languages[j]:
		hindiPOSCount.append(len(eval(intPOSTags[j])))
	if "SPA" in Languages[j]:
		spanishPOSCount.append(len(eval(intPOSTags[j])))
	if "ITA" in Languages[j]:
		italianPOSCount.append(len(eval(intPOSTags[j])))
	if "JPN" in Languages[j]:
		japanesePOSCount.append(len(eval(intPOSTags[j])))
	if "TEL" in Languages[j]:
		teluguPOSCount.append(len(eval(intPOSTags[j])))

languagesString = str(set(Languages))
print("Essays collected: " + str(len(POSTags)))
print("Languages collected: " + str(len(Languages)))
print("Number of unique Part of Speech Tags: " + str(len(uniqueTags)))
print("Languages surveyed: " + re.sub(r'\\n|{|}', ' ', languagesString))
print("Number of Unique Part of Speech Tags: " + str(len(set(POSTags))))

print("Number of responses by first language: " +
	"\n\t German: \t{} \n\t Turkish: \t{} \n\t French: \t{} \n\t Arabic: \t{} \n\t Korean: \t{} \n\t Chinese: \t{}\n\t Hindi: \t{} \n\t Spanish: \t{} \n\t Italian: \t{} \n\t Japanese: \t{} \n\t Telugu: \t{}"
	.format(len(germanPOSCount), len(turkishPOSCount), len(frenchPOSCount),len(arabicPOSCount),len(koreanPOSCount), len(chinesePOSCount), len(hindiPOSCount), len(spanishPOSCount), len(italianPOSCount), len(japanesePOSCount), len(teluguPOSCount)))
print("Average number of POS tags in each essay by first language: " +
	"\n\t German: \t{} \n\t Turkish: \t{} \n\t French: \t{} \n\t Arabic: \t{} \n\t Korean: \t{} \n\t Chinese: \t{}\n\t Hindi: \t{} \n\t Spanish: \t{} \n\t Italian: \t{} \n\t Japanese: \t{} \n\t Telugu: \t{}"
	.format(sum(germanPOSCount)/len(germanPOSCount), sum(turkishPOSCount)/len(turkishPOSCount), sum(frenchPOSCount)/len(frenchPOSCount),sum(arabicPOSCount)/len(arabicPOSCount),
		sum(koreanPOSCount)/len(koreanPOSCount), sum(chinesePOSCount)/len(chinesePOSCount), sum(hindiPOSCount)/len(hindiPOSCount), sum(spanishPOSCount)/len(spanishPOSCount), sum(italianPOSCount)/len(italianPOSCount), sum(japanesePOSCount)/len(japanesePOSCount),
		sum(teluguPOSCount)/ len(teluguPOSCount)))
allArrays = germanPOSCount + turkishPOSCount +frenchPOSCount +arabicPOSCount +koreanPOSCount +chinesePOSCount +hindiPOSCount +spanishPOSCount +italianPOSCount +japanesePOSCount + teluguPOSCount 

h = sorted(allArrays)
fit = stats.norm.pdf(h, np.mean(h), np.std(h), )
pl.plot(h, fit, '-o', color = "orange")
pl.hist(h, 100, normed = True, facecolor = 'blue')
pl.xlabel('Length of Responses', fontsize = 30)
pl.ylabel('Number of Responses',fontsize = 30)
pl.show()