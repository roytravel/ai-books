# %% set hyperparameter
import torch
from ratsnlp.nlpbook.classification import ClassificationTrainArguments

args = ClassificationTrainArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_task_name="pair-classification",
    downstream_corpus_name="klue-nli",
    downstream_model_dir="nlpbook/checkpoint-paircls",
    batch_size=32 if torch.cuda.is_available() else 4,
    learning_rate=5e-5,
    max_seq_length=128,
    epochs=5,
    tpu_cores=0,
    seed=7)


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
    pretrained_model_name_or_path=args.pretrained_model_name,
    do_lower_case=False,
)
# %% create train data
from ratsnlp.nlpbook.paircls import KlueNLICorpus
from ratsnlp.nlpbook.classification import ClassificationDataset
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

corpus = KlueNLICorpus()
train_dataset = ClassificationDataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="train",
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler = RandomSampler(train_dataset, replacement=False),
    collate_fn=nlpbook.data_collator,
    drop_last=False,
    num_workers=0,
)

# %% create test dataset
val_dataset = ClassificationDataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="test",
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    sampler = SequentialSampler(val_dataset),
    collate_fn=nlpbook.data_collator,
    drop_last=False,
    num_workers=0,
)
# %% initialize model

from transformers import BertConfig, BertForSequenceClassification

pt_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
    num_labels=corpus.num_labels,
)

model = BertForSequenceClassification.from_pretrained(
    args.pretrained_model_name,
    config=pt_model_config,
)

# %% prepare training
from ratsnlp.nlpbook.classification import ClassificationTask
task = ClassificationTask(model, args)

# %%
trainer = nlpbook.get_trainer(args)

# %%
trainer.fit(
    task, 
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
)

# %%
