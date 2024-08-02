The dataset contains Named Entity and Parts of Speech tags of about 7000 randomly selected Bangla sentences.
 The tags are automatically genertaed using a inhouse pre-trained model.

The dataset is formated in the following format:

sentence_1		
token_1	pos_tag	ner_tag
token_2	pos_tag	ner_tag
token_3	pos_tag	ner_tag
		
sentence_2		
token_1	pos_tag	ner_tag
token_2	pos_tag	ner_tag
token_3	pos_tag	ner_tag


There are 16 unique POS tags:
NNP: Proper Noun, Singular
PUNCT: Punctuation
NNC: Common Noun
ADJ: Adjective
DET: Determiner
VF: Finite Verb
CONJ: Conjunction
PRO: Pronoun
VNF: Non-finite Verb
PP: Postposition
QF: Quantifier
ADV: Adverb
PART: Particle
OTH: Other
INTJ: Interjection
Invalid Tag: Invalid Tag




There are 21 unique NER Tags:
B-D&T: Beginning of Date and Time
I-D&T: Inside Date and Time
B-OTH: Beginning of Other
B-GPE: Beginning of Geo-Political Entity
I-GPE: Inside Geo-Political Entity
B-PER: Beginning of Person
I-PER: Inside Person
B-LOC: Beginning of Location
I-LOC: Inside Location
B-ORG: Beginning of Organization
I-ORG: Inside Organization
B-EVENT: Beginning of Event
I-EVENT: Inside Event
B-NUM: Beginning of Number
I-NUM: Inside Number
B-UNIT: Beginning of Unit
I-UNIT: Inside Unit
B-MISC: Beginning of Miscellaneous
I-MISC: Inside Miscellaneous
B-T&T: Beginning of Title and Term
I-T&T: Inside Title and Term
