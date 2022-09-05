# %%
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

# %% 
porter_stemmer = PorterStemmer()
snowball_stemmer = SnowballStemmer("english")

print(porter_stemmer.stem("fastest")) # fastest
print(snowball_stemmer.stem("fastest")) # fastest
# %%
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
lemmatizer('fastest', 'ADJ') # ['fast']

# %% 
"RESULT: Lemmatization is more accurate than Stemmization."
