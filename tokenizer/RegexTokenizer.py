from BaseTokenizer import Tokenizer, get_stats, merge
import regex as re

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN
        self.pat_compile = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}


    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256 # 256 - number of integers

        # split text into chunks
        chuncks = re.findall(self.pat_compile, text)

        ids = [list(ch.encode("utf-8")) for ch in chuncks]


        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)
            
            pair = max(stats, key=stats.get)
            idx = 256 + i

            # replace all apperances of pair ids with idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]

            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose: print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")


        self.merges = merges
        self.vocab = vocab

    def register_speacial_tokens(self, special_tokens):
        # special tokens like <|endoftext|>
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
    
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
    regex_token = RegexTokenizer()
    # open some text and train a vocab of 512 tokens
    text = open("tests/taylor.txt", "r", encoding="utf-8").read()
    vocab_size = 512

    # train BasicTokenizer
    regex_token.train(text, vocab_size, verbose=True)

    # Encode some text
    encoded_text = regex_token.encode("Hello World!")
    print(encoded_text)

    # Decode some Integers
    # decoded_text = regex_token.decode([72, 101, 301, 111, 346, 258, 509, 33])
    # print(decoded_text)

    # Test on all taylor text
    # print(regex_token.decode(regex_token.encode(text)) == text)

