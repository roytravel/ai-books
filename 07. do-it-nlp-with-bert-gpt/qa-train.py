# %% set hyperparameters
import torch
from ratsnlp.nlpbook.qa import QATrainArguments
args = QATrainArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_corpus_name="korquad-v1",
    downstream_model_dir="nlpbook/checkpoint-qa",
    max_seq_length=128,
    max_query_length=32,
    doc_stride=64,
    batch_size=32 if torch.cuda.is_available() else 4,
    learning_rate=5e-5,
    epochs=3,
    tpu_cores=0 if torch.cuda.is_available() else 8,
    seed=7,
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
from ratsnlp.nlpbook.qa import KorQuADV1Corpus, QADataset
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
corpus = KorQuADV1Corpus()
train_dataset = QADataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="train",
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=RandomSampler(train_dataset, replacement=False),
    collate_fn=nlpbook.data_collator,
    drop_last=False,
    num_workers=args.cpu_workers,
)
# %% create valid dataset
val_dataset = QADataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="val",
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    sampler=SequentialSampler(val_dataset),
    collate_fn=nlpbook.data_collator,
    drop_last=False,
    num_workers=args.cpu_workers,
)