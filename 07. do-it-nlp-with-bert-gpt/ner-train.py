# %% set hyperparameter

import torch
from ratsnlp.nlpbook.ner import NERTrainArguments

args = NERTrainArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_corpus_name="ner",
    downstream_model_dir="nlpbook/checkpoint-ner",
    batch_size = 64 if torch.cuda.is_available() else 4,
    learning_rate=5e-5,
    max_seq_length=64, 
    epochs=3,
    tpu_cores = 0,
    seed = 7,
)

# %% set random seed
from ratsnlp import nlpbook
nlpbook.set_seed(args)

# %% set logger
nlpbook.set_logger(args)

# %% download corpus
nlpbook.download_downstream_dataset(args)

# %% prepare tokenizer
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
)

# %% create train dataset
from ratsnlp.nlpbook.ner import NERCorpus, NERDataset
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

corpus = NERCorpus(args)
train_dataset = NERDataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="train"
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler = RandomSampler(train_dataset, replacement=False),
    collate_fn = nlpbook.data_collator,
    drop_last=False,
    num_workers = 0
)

# %% 
print (train_dataset)

# %% create test dataset
val_dataset = NERDataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="val",
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    sampler=SequentialSampler(val_dataset),
    num_workers=0,
    collate_fn=nlpbook.data_collator,
    drop_last=False,
)

# %% initialize model
from transformers import BertConfig
pt_model_confog = BertConfig.from_pretrained(
    args.pretrained_model_name,
    num_labels = corpus.num_labels,
)
# %%
from transformers import BertForTokenClassification
model = BertForTokenClassification.from_pretrained(
    args.pretrained_model_name,
    config=pt_model_confog
)
# %% prepare training
from ratsnlp.nlpbook.ner import NERTask
task = NERTask(model, args)

# %%
trainer = nlpbook.get_trainer(args)
# %%
trainer.fit(
    task,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader
)
# %%
