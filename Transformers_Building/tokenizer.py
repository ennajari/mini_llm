class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
    def encode(self, s): return [self.stoi[c] for c in s]
    def decode(self, t): return ''.join([self.itos[i] for i in t])
