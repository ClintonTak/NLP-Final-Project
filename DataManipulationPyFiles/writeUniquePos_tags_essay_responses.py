#find all the unique POS tags in there and make a file so that it looks like the "CaesTags.txt" file

input_file = open('../Data/TOEFL/POS_tags_essay_responses.txt', 'r')
f = open('../Data/TOEFL/Pos_Tags.txt','w')
for line in input_file:
	posTags = []
	for item in line.split():
		if item not in posTags:
			posTags.append(item)
	
for tag in posTags:
	f.write(tag)
	f.write("\n")
f.close()
input_file.close()
