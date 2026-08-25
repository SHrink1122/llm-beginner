# tokenizer.py
# 字符级tokenizer
import torch
from torch.utils.data import Dataset

PAD_ID, UNK_ID, CLS_ID = 0, 1, 2
MAX_SEQ_LEN = 256
class CharTokenizer:
    def __init__(self, chars, max_len=MAX_SEQ_LEN):
        self.max_len = max_len
        self.char2id = {ch: i + 3 for i, ch in enumerate(chars)}
        self.id2char = {i + 3 : ch for i, ch in enumerate(chars)}

        self.id2char[PAD_ID] = "<pad>"
        self.id2char[UNK_ID] = "<unk>"
        self.id2char[CLS_ID] = "<cls>"

    @property
    def vocab_size(self):
        return 3 + len(self.char2id)

    def encode(self, text):
        ids = [self.char2id.get(ch, UNK_ID) for ch in text[: self.max_len]]
        return ids

    def tokenize(self, text):
        return torch.tensor(self.encode(text), dtype=torch.long)

def build_tokenizer(texts, max_len=MAX_SEQ_LEN):
    chars = sorted({ch for text in texts for ch in text})
    return CharTokenizer(chars, max_len=MAX_SEQ_LEN)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.ids = [tokenizer.encode(t) for t in texts]
        self.labels = list(labels)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return torch.tensor(self.ids[i], dtype=torch.long), self.labels[i]
    
def collate_fn(batch):
    ids, labels = zip(*batch)
    max_len = max(x.size(0) for x in ids)
    padded = torch.zeros(len(ids), max_len, dtype=torch.long)
    for i, x in enumerate(ids):
        padded[i, : x.size(0)] = x
    return padded, torch.tensor(labels, dtype=torch.long)