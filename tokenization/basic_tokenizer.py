from base import Tokenizer
from utils import get_stats, merge


class BasicTokenizer(Tokenizer):
    """
    Minimal (byte-level) Byte Pair Encoding tokenizer.

    Algorithmically follows along the GPT tokenizer:
    https://github.com/openai/gpt-2/blob/master/src/encoder.py

    But:
    - Does not handle the regular expression splitting pattern.
    - Does not handle any special tokens.
    """

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        # train a vocab base on input text
        # vocab size have to greater or equal than 256 since in original utf-8 encode will have 255 int token
        assert vocab_size >= 256
        # calculate how many times merge needed
        num_merges = vocab_size - 256

        # input text preprocessing, convert to raw bytes
        text_bytes = text.encode("utf-8")
        # list of integers in range 0..255
        ids = list(text_bytes)

        # iteratively merge the most common pairs to create new tokens
        # (int, int) -> int, ex: (101, 32): 256
        merges = {}
        # int -> bytes, map token index into bytes string
        # inside gpt2, actually we have 50257 tokens, 256 is raw bytes token, 50000 is merged token, 1 is special token
        vocab = {idx: bytes([idx]) for idx in range(256)}
        # loop through all merge times
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            # ex: (1, 2): 2, pair is (1, 2) with the highest count 2
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            # ex: (101, 32): 256, means pair (101, 32) merged into new ids 256
            merges[pair] = idx
            # update vocab as well
            # ex: 256: bytes([101]) + bytes([32]) -> b'e ' (bytes string)
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # if print
            if verbose:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        # used in encode()
        self.merges = merges
        # used in decode()
        self.vocab = vocab

    def decode(self, ids):
        # given ids (list of integers), return Python string
        # join list of bytes string according to vocab
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        # convert bytes string into string, if face the error, will be replaced by special character like '?'
        text = text_bytes.decode("utf-8", errors="replace")

        return text

    def encode(self, text):
        # given a string text, return the token ids
        # convert to raw bytes first, ex:
        text_bytes = text.encode("utf-8")
        # list of integers in range [0, 255]
        ids = list(text_bytes)
        # merge until no more pair shows up inside merges dictionary or new ids length last than 2 (a pair)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            # get pair with the frequency first
            stats = get_stats(ids)
            # get the lowest merge index according to training defined merge dictionary
            # the lambda function will get all pair new_idx in merges dictionary, if not exist will be float("inf")
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            # in here we check if nothing else can be merged anymore
            if pair not in self.merges:
                break
            # otherwise let's merge the best pair (lowest merge index)
            # get new merged idx
            idx = self.merges[pair]
            # merge ids list into new ids according to new idx
            ids = merge(ids, pair, idx)

        return ids
