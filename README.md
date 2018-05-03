# NLP-Final-Projects
Collaborators: 

* [Daniel Brett](https://github.com/dbrett90)
* [Lauren Becker](https://github.com/lnbecker)
* [Clinton Tak](https://github.com/clintontak)
* [Isaiah Rawlinson](https://github.com/irawlinson)

## Project Scope and Goals

Using data from the CAES institute, we want to make a [native language inference/identification](https://en.wikipedia.org/wiki/Native-language_identification) system that classifies a persons native language based on how they write a second language. We are using data from a Spanish Language test, with participants that spoke Chinese, Portuguese, Russian, French, English, and Arabic. We then use data from the COEFL which contains a larger corpus of essays (written in english) with a wider diversity of native speakers. These languages include German, Turkish, French, Arabic, Korean, Chinese, Hindi, Spanish, Italian, Japanese, and Telugu. Information about each of these is outlined in the following sections. 

## Metadata and Associated Information

**General Essay Information (CAES)**


| Essay Category      | Associated Number       | 
| ------------- |:-------------:| 
| Respondents   | 3878 | 
| Essay Samples | 3878   |  
| Total POS Tags | 682172     | 
|Average POS tags per essay | 175.9| 

**Responses and Tags by Language (CAES)**


| Language | Total Essays  | Average POS Tags per Essay|
| ------------- |:-------------:| :-----:|
| Arabic  | 1342| 148.3 |
 | Chinese | 373     |   169.2 |
| English | 615 |   204.7|
| French  | 371 | 189.5 |
| Russian | 176 | 140.8|
| Portuguese| 1001 | 198.8|



## Challenges 

Unfortunately, the data that we have is not very rich. We are working with essays that have been transcribed into part of speech tags meaning we are not utilizing the raw essays. This makes working with the text easier because it is standardized, but limits the amount of data that we can gather from the text as the phrases have been completely normalized. 

## Applications 

One application of NLI is its use in forensic linguistics. With the rise of Russian troll farms, being able to accurately determine which texts were written by native language speakers versus which texts were written by native Russian speakers would turn the tide of misinformation and propaganda that has flooded the internet. In addition, a number of intelligence agencies have started to [fund NLI projects](https://research.aston.ac.uk/portal/en/theses/linguistic-identifiers-of-l1-persian-speakers-writing-in-english(4e21bce7-f3af-47ec-8101-971a9f20b436).html) in the hopes that it will give them more information about potential threats and who are responsible for them. NLI also has applications as it pertains to pedagogical (teaching) materials. By identifying L1-specific features, we can improve language transfer and author profiling.

## Formal Report 

A PDF file containing all relevant background and findings from this project (along with associated literature reviews) can be found within the [reportDocs folder](https://github.com/ClintonTak/NLP-Final-Projects/tree/master/reportDocs). 

## Other work

This work is based on research from a few shared tasks. Here are links to further reading:

* [A Report on the 2017 Native Language Identification Shared Task](https://www.aclweb.org/anthology/W/W17/W17-5007.pdf)

* [A Report on the First Native Language Identification Shared Task](http://www.aclweb.org/anthology/W13-1706)

* [Feature Analysis for Native Language Identification](http://nlp.unibuc.ro/papers/nisioi15a.pdf) 

  ​
