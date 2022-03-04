import os
import ratsnlp
from Korpora import Korpora
from tokenizers import ByteLevelBPETokenizer
from tokenizers import BertWordPieceTokenizer

class CREATE_TOKENIZER():
    def __init__(self) -> None:
        self.nsmc = Korpora.load("nsmc", force_download=True)
        os.makedirs("nlpbook/bbpe", exist_ok=True)
        os.makedirs("nlpbook/wordpiece", exist_ok=True)
        
    def write_lines(self, path, lines) -> None:
        with open(path, mode='w', encoding='utf-8') as f:
            for line in lines:
                f.write(f'{line}\n')
    
    def save_dataset(self) -> None:
        self.write_lines("./train.txt", self.nsmc.train.get_all_texts())
        self.write_lines("./test.txt", self.nsmc.test.get_all_texts())

    def train_BPE_tokenizer(self) -> None:
        bytebpe_tokenizer = ByteLevelBPETokenizer()
        bytebpe_tokenizer.train(
            files=['./train.txt', './test.txt'],
            vocab_size = 10000,
            special_tokens=["[PAD]"])

        bytebpe_tokenizer.save_model("nlpbook/bbpe")

    def train_wordpiece_tokenizer(self) -> None:
        wordpiece_tokenizer = BertWordPieceTokenizer()
        wordpiece_tokenizer.train(
            files=["./train.txt", "./test.txt"],
            vocab_size = 10000,)
        
        wordpiece_tokenizer.save_model("nlpbook/wordpiece")
        

if __name__ == "__main__":
    T = CREATE_TOKENIZER()    
    T.save_dataset()
    T.train_BPE_tokenizer()
    T.train_wordpiece_tokenizer()
