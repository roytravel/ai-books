# %% 
import spacy
nlp = spacy.load('en_core_web_sm')

# %% example 1
string = u"Google has its headquarters in Mountain View, California having revenue amounted to 109.65 billion US dollars"
doc = nlp(string)

for ent in doc.ents:
    print (ent.text, ent.label_)
# %% example 2
string = u"Mark Zuckerberg born May 14, 1984 in New York is an american technology entrepreneur and philanthropist best known for co-founding and leading Facebook as its chairman and CEO."
doc = nlp(string)

for ent in doc.ents:
    print(ent.text, ent.label_)

# %% example 3
string = u"I usually wake up at 9:00 AM. 90% of my daytime goes in learning new things."
doc = nlp(string)
for ent in doc.ents:
    print (ent.text, ent.label_)

# %%
string1 = u"Imagine Dragons are the best band"
string2 = u"Imagine dragons come and take over the city"

doc1 = nlp(string1)
doc2 = nlp(string2)

# %% 
for ent in doc1.ents:
    print (ent.text, ent.label_)
# %% 
for ent in doc2.ents:
    print (ent.text, ent.label_) # there is no result.
