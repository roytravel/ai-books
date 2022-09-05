# %% 
import spacy
nlp = spacy.load("en_core_web_sm")

# %%
doc = nlp(u"How are you doing today?")
for token in doc:
    print (token.text, token.vector[:5])
# %%

hello_doc = nlp(u"hello")
hi_doc = nlp(u"hi")
hella_doc = nlp(u"hella")

print(hello_doc.similarity(hi_doc)) # more similar than hella.
print(hello_doc.similarity(hella_doc))
# %%
str1 = nlp(u"When will next season of Game of Thrones be releasing?")
str2 = nlp(u"Game of Thrones next season release date?")
str1.similarity(str2)
# %%

example = nlp(u"car truck google")

for t1 in example:
    for t2 in example:
        similarity_perc = int(t1.similarity(t2) * 100)
        print (f"Word {t1.text} is {similarity_perc} to word {t2.text}")
# %%
