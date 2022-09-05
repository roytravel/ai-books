""" The first thing when we make chatbot is anlayzing what kind of part is exist in sentence. """
# %% 
import spacy
nlp = spacy.load('en_core_web_sm')

# %%  example 1
doc = nlp(u"I am learning how to build chatbots")
for token in doc:
    print (token.text, token.pos_)

# %% example 2
doc = nlp(u"I am going to London next week for a meeting)")
for token in doc:
    print (token.text, token.pos_)

# %% example 3
doc = nlp(u"Google release 'Move Mirror' AI experiment that matches your pose from 80,000 images")
for token in doc:
    print (token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)

# %% example 4
doc = nlp(u"I am learning how to build chatbots")
for token in doc:
    print (token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)