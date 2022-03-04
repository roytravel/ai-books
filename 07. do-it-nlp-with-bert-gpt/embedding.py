# %% Load library
import torch
from transformers import BertTokenizer, BertConfig, BertModel

# %% Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base", do_lower_case=False)

# %% Initialize model
pt_model_config = BertConfig.from_pretrained("beomi/kcbert-base")
model = BertModel.from_pretrained("beomi/kcbert-base", config=pt_model_config)

# %% check configuration
print (pt_model_config)

# %% create an input value for input
sentences = ["안녕하세요", "반갑습니다", "저의 이름은", "로이입니다"]
features = tokenizer(
    sentences,
    max_length=10,
    padding = "max_length",
    truncation=True,
)


# %%
print (features.keys())
# %%
print (features['input_ids'])
# %%
print (features['token_type_ids'])
# %%
print (features['attention_mask'])

# %% extract embedding of bert
feats = {}
for k, v in features.items():
    feats[k] = torch.tensor(v)
# %%
print (feats)
# %%
outputs = model(**feats)

# %% 
print (outputs.last_hidden_state.shape)
# %%
print (outputs.pooler_output)
# %%
print (outputs.pooler_output.shape)
# %%

print ((outputs.pooler_output))
# %%

# %%

# %%
