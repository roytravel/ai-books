# Reference: bit.ly/3FjJJ1H
# %% load library
import torch
from Korpora import Korpora
from transformers import BertTokenizer
from transformers import BertConfig, BertForSequenceClassification
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

from ratsnlp import nlpbook
from ratsnlp.nlpbook.classification import ClassificationTrainArguments, ClassificationTask
from ratsnlp.nlpbook.classification import NsmcCorpus, ClassificationDataset

# %% set hyperameters
args = ClassificationTrainArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_corpus_name="nsmc",
    downstream_model_dir="nlpbook/checkpoint-doccls",
    batch_size=32 if torch.cuda.is_available() else 4,
    learning_rate=5e-5,
    max_seq_length=128,
    epochs=3,
    #tpu_cores=0 if torch.cuda.is_available() else 8,
    tpu_cores=0,
    seed=7,
)

# %% set random seed
nlpbook.set_seed(args)

# %% set logger
nlpbook.set_logger(args)

# %% download corpus
Korpora.fetch(
    corpus_name=args.downstream_corpus_name,
    root_dir=args.downstream_corpus_root_dir,
    force_download=True,
)

# %% Tokenizer
tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
)

# %% create train dataset
corpus = NsmcCorpus()
train_dataset = ClassificationDataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="train"
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=RandomSampler(train_dataset, replacement=False),
    collate_fn=nlpbook.data_collator,
    drop_last = False,
    #num_workers=args.cpu_workers,
    num_workers=0
)

# %% create test dataset
val_dataset = ClassificationDataset(
    args=args,
    tokenizer=tokenizer,
    corpus=corpus,
    mode="test",
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    sampler=SequentialSampler(val_dataset),
    collate_fn=nlpbook.data_collator,
    drop_last=False,
    num_workers=args.cpu_workers,
)

# %% initialize model
pt_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
    num_labels=corpus.num_labels,
)

model = BertForSequenceClassification.from_pretrained(
    args.pretrained_model_name,
    config=pt_model_config,
)


# %% prepare training
task = ClassificationTask(model, args)

trainer = nlpbook.get_trainer(args)

# %% training

trainer.fit(
    task,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
)
