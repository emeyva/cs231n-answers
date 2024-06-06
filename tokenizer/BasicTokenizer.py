"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from BaseTokenizer import Tokenizer, merge, get_stats

class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256 # 256 - number of integers
        text_bytes = text.encode("utf-8")
        tokens = list(map(int,text_bytes))
        ids = list(tokens)
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            if verbose: print(f"merging {pair} into a new token {idx}")
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        self.merges = merges
        self.vocab = vocab
    
    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # nothing else can be merged
            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)
        return tokens

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

if __name__ == '__main__':
    basic_token = BasicTokenizer()
    # open some text and train a vocab of 512 tokens
    text = open("tests/taylor.txt", "r", encoding="utf-8").read()
    vocab_size = 512

    # train BasicTokenizer
    basic_token.train(text, vocab_size, verbose=False)

    # Encode some text
    encoded_text = basic_token.encode("Hello World!")
    print(encoded_text)

    # Decode some Integers
    decoded_text = basic_token.decode([72, 101, 301, 369, 87, 291, 108, 100, 33])
    print(decoded_text)

    # Test on all taylor text
    print(basic_token.decode(basic_token.encode(text)) == text)

