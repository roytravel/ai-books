# %% 
import spacy
nlp = spacy.load('en_core_web_sm')

# %% 
doc = nlp(u"Boston Dynamics is gearing up to produce thousands of robot dogs")
list(doc.noun_chunks)

# %%
doc = nlp(u"Deep learning cracks the code of messenger RNAs and protein-coding potential")
for chunk in doc.noun_chunks:
    print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)
# %%