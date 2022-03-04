from transformers import GPT2Tokenizer

tokenizer_gpt = GPT2Tokenizer.from_pretrained("nlpbook/bbpe")
tokenizer_gpt.add_special_tokens({'pad_token': '[PAD]'})

sentences = [
    "아 더빙.. 진짜 짜증나네요 목소리",
    "흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나",
    "별루 였다..",
]

gpt_tokenized_sents = []

for sentence in sentences:
    gpt_tokenized_sents.append(tokenizer_gpt.tokenize(sentence))
    
print (gpt_tokenized_sents)

batch_inputs = tokenizer_gpt(
    sentences, 
    padding="max_length",
    max_length=12,
    truncation=True, # truncation if count of token is over max_length
)

print (batch_inputs.keys())

print (batch_inputs['input_ids'])
print (batch_inputs['attention_mask'], end='\n\n')

from transformers import BertTokenizer

tokenizer_bert = BertTokenizer.from_pretrained("nlpbook/wordpiece")

bert_tokenized_sents = []
for sentence in sentences:
    bert_tokenized_sents.append(tokenizer_bert.tokenize(sentence))

print (bert_tokenized_sents)

batch_inputs = tokenizer_bert(
    sentences,
    padding="max_length",
    max_length=12,
    truncation=True,
)

print (batch_inputs.keys())
print (batch_inputs['input_ids'])
print (batch_inputs['token_type_ids'])
print (batch_inputs['attention_mask'])