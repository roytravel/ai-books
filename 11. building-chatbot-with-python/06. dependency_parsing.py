# %% load library
import spacy
nlp = spacy.load('en_core_web_sm')

# %% 
doc = nlp(u"Book me a flight from Bangalore to Goa")
blr, goa = doc[5], doc[7]

# %% 
list(blr.ancestors)
# %%
list(goa.ancestors)
# %%
list(doc[4].ancestors)
# %%
doc[3].is_ancestor(doc[5]) # flight, Bangalore
# %% case 1

doc = nlp(u"Book a table at the restaurant and the taxi to the hotel")
tasks = doc[2], doc[8] # (table, taxi)
tasks_target = doc[5], doc[11] # (restaurant, hotel)

for task in tasks_target:
    for tok in task.ancestors:
        if tok in tasks:
            print (f"Booking of {tok} belongs to {task}")

# %% 
# from spacy import displacy
# doc = nlp(u"Book a table at the at the restaurant and the taxi to the hotel")
# displacy.serve(doc, style='dep')
doc = nlp(u"What are some places to visit in Berlin and stay in Lubeck")
places = [doc[7], doc[11]]
actions = [doc[5], doc[9]]

for place in places:
    for tok in place.ancestors:
        if tok in actions:
            print (f"User is referring {place} to {tok}")
