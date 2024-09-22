from utils import render_token


class Tokenizer:
    """
    Contains the base Tokenizer class.
    The base class also contains the (common) save/load functionality.
    It would be possible to be a lot stricter about the interface and
    e.g. isolating all regex/pattern parts to the RegexTokenizer, but
    some concessions are made for simplicity.
    """

    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        # (int, int) -> int, ex: (101, 32): 256
        self.merges = {}
        # str, represent split pattern for regex tokenization
        self.pattern = ""
        # str -> int, e.g. {'<|endoftext|>': 100257}
        self.special_tokens = {}
        # map int -> into bytes string
        self.vocab = self._build_vocab()

    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text):
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        # map token index into bytes string, ex: {0: b'\x00', 1: b'\x01', 2: b'\x02', 3: b'\x03'}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        # for each merged index, convert this merged index by combine two other bytes string
        for (p0, p1), idx in self.merges.items():
            # ex: 479: b'January ' in vocab
            vocab[idx] = vocab[p0] + vocab[p1]

        # map special token idx to raw bytes string
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")

        return vocab

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # write the merges dict, only save merged pair index, ex: pair -> {(100, 32): 258}, idx1: 100, idx2: 32
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        # inverse the merge dictionary
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """
        Inverse of save() but only for the model file
        """
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read and build the merges, ex: {(101, 32): 256}
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
