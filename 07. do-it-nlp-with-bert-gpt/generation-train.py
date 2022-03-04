# %% set hyperparameter
from lib2to3.pgen2 import token
from random import Random
from grpc import GenericRpcHandler
import torch
from ratsnlp.nlpbook.generation import GenerationTrainArguments

args = GenerationTrainArguments(
    pretrained_model_name="skt/kogpt2-base-v2",
    downstream_corpus_name="nsmc",
    downstream_model_dir="nlpbook/checkpoint-generation",
    max_seq_length=32,
    batch_size= 64 if torch.cuda.is_available() else 4,
    learning_rate=5e-5,
    epochs=3,
    tpu_cores = 0,
    seed=7,
)


# %% set random seed
from ratsnlp import nlpbook
nlpbook.set_seed(args)

# %% set logger

nlpbook.set_logger(args)

# %% download corpus
from Korpora import Korpora

Korpora.fetch(
    args.downstream_corpus_name,
    root_dir=args.downstream_corpus_root_dir,
    force_download=args.force_download,
    )

# %% prepare tokenizer
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    args.pretrained_model_name,
    eos_token="</s>",
)

# %% create train dataset
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from ratsnlp.nlpbook.generation import GenerationDataset, NsmcCorpus

corpus = NsmcCorpus()

train_dataset = GenerationDataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="train"
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler = RandomSampler(train_dataset, replacement=False),
    collate_fn=nlpbook.data_collator,
    drop_last=False,
    num_workers=0,
)

# %% create validation dataset

val_dataset = GenerationDataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="test",
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    sampler=SequentialSampler(val_dataset),
    collate_fn=nlpbook.data_collator,
    drop_last = False,
    num_workers=0,
)
# %% initialize model
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained(
    args.pretrained_model_name
)

# %% prepare training
from ratsnlp.nlpbook.generation import GenerationTask
task = GenerationTask(model, args)

# %%
trainer = nlpbook.get_trainer(args)
# %% train
trainer.fit(
    task,
    train_dataloader = train_dataloader,
    val_dataloaders = val_dataloader
)

# %%
