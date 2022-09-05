# %%
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)

# %% 
lemmatizer('chuckles', 'NOUN') # ['chunkle']
# %%
lemmatizer('blazing', 'VERB') # ['blaze']

# %%
lemmatizer('fastest', 'ADJ') # ['fast']