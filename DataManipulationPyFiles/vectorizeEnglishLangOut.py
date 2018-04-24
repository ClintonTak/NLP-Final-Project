#converts language strings in EnglishLangOut.txt to one hot vectors in alphabetical order 
#[Arabic, Chinese, English, French, Portuguese, Russian]

input_file = open('../Data/EnglishLangOut.txt', 'r')
f = open('../Data/EnglishLangOutVectors.txt','w')
for line in input_file:
	if "Arabic" in line:
		f.write("[1, 0, 0, 0, 0, 0]\n")
	if "Chinese" in line:
		f.write("[0, 1, 0, 0, 0, 0]\n")
	if "English" in line:
		f.write("[0, 0, 1, 0, 0, 0]\n")
	if "French" in line:
		f.write("[0, 0, 0, 1, 0, 0]\n")
	if "Portuguese" in line:
		f.write("[0, 0, 0, 0, 1, 0]\n")
	if "Russian" in line:
		f.write("[0, 0, 0, 0, 0, 1]\n")
f.close()
