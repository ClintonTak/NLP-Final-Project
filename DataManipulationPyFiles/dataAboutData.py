import re
cAESTagsFiles = open('../Data/CaesTags.txt', 'r')
languagesFiles = open('../Data/EnglishLangOut.txt', 'r')
vectorFiles = open('../Data/EnglishLangOutVectors.txt','r')
posTagsFiles = open('../Data/tags.txt', 'r')
intPOSTagsFiles = open('../Data/vectorTags.txt')
individualLanguages = languagesFiles.readlines()
vectors = vectorFiles.readlines()
posTags = posTagsFiles.readlines()
intPOSTags = intPOSTagsFiles.readlines()
caesTags = cAESTagsFiles.readlines()

totalPOSTags = 0
##getting total number of POS tags
for i in intPOSTags:
	totalPOSTags += len(eval(i))
##getting average number of POS tags by language 
arabicPOSCount = []
chinesePOSCount = []
englishPOSCount = []
frenchPOSCount = []
russianPOSCount = []
portuguesePOSCount = []
for j in range(0, len(individualLanguages)):
	if "Arabic" in individualLanguages[j]:
		arabicPOSCount.append(len(eval(intPOSTags[j])))
	if "Chinese" in individualLanguages[j]:
		chinesePOSCount.append(len(eval(intPOSTags[j])))
	if "English" in individualLanguages[j]:
		englishPOSCount.append(len(eval(intPOSTags[j])))
	if "French" in individualLanguages[j]:
		frenchPOSCount.append(len(eval(intPOSTags[j])))
	if "Russian" in individualLanguages[j]:
		russianPOSCount.append(len(eval(intPOSTags[j])))
	if "Portuguese" in individualLanguages[j]:
		portuguesePOSCount.append(len(eval(intPOSTags[j])))
		
languagesString = str(set(individualLanguages))



print("Number of respondents: " + str(len(vectors)))
print("Number of essay samples: " + str(len(posTags)))
print("Number of unique part of speech tags: " + str(len(caesTags)))
print("Languages of respondents: " + re.sub(r'\\n|{|}', ' ', languagesString))
print("Total part of speech tags: " + str(totalPOSTags))
print("Average number of part of speech tags per essay: " +str(totalPOSTags/len(posTags)))
print("Number of responses by first language: " +
	"\n\t Arabic: \t{} \n\t Chinese: \t{} \n\t English: \t{} \n\t French: \t{} \n\t Russian: \t{}"
	.format(len(arabicPOSCount), len(chinesePOSCount), len(englishPOSCount),len(frenchPOSCount),len(russianPOSCount), len(portuguesePOSCount)))
print("Average number of POS tags in each essay by first language: " +
	"\n\t Arabic: \t{} \n\t Chinese: \t{} \n\t English: \t{} \n\t French: \t{} \n\t Russian: \t{}"
	.format(sum(arabicPOSCount)/len(arabicPOSCount), sum(chinesePOSCount)/len(chinesePOSCount), 
		sum(englishPOSCount)/len(englishPOSCount), sum(frenchPOSCount)/len(frenchPOSCount), sum(russianPOSCount)/len(russianPOSCount), sum(portuguesePOSCount)/len(portuguesePOSCount)))
