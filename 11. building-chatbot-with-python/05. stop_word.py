# %% load library
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# %% load 
nlp = spacy.load('en_core_web_sm')

# check stop word.
print(STOP_WORDS)
# %% example 1
nlp.vocab[u'is'].is_stop # True
# %%
nlp.vocab[u'hello'].is_stop # False

# %%
nlp.vocab[u'with'].is_stop