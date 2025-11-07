import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, path, tokenizer_src, tokenizer_tgt, block_size=32):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if '\t' not in line:
                    continue
                src, tgt = line.strip().split("\t", 1)
                self.samples.append((src, tgt))
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.block_size = block_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt = self.samples[idx]
        x = self.tokenizer_src.encode(src)[:self.block_size]
        y = self.tokenizer_tgt.encode(tgt)[:self.block_size]
        if len(x) < self.block_size:
            x = x + [0]*(self.block_size - len(x))
        if len(y) < self.block_size:
            y = y + [0]*(self.block_size - len(y))
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return x, y
