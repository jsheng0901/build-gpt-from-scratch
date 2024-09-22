import regex as re
from base import Tokenizer
from utils import get_stats, merge


class RegexTokenizer(Tokenizer):
    """
    Minimal (byte-level) Byte Pair Encoding tokenizer.

    Algorithmically follows along the GPT tokenizer:
    https://github.com/openai/gpt-2/blob/master/src/encoder.py

    Unlike BasicTokenizer:
    - RegexTokenizer handles an optional regex splitting pattern.
    - RegexTokenizer handles optional special tokens.
    """

    # the main GPT text split patterns, means when string meet this patterns which will be split
    # see https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
    GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        # pattern regex string used in split text, default is GPT4
        self.pattern = self.GPT4_SPLIT_PATTERN if pattern is None else pattern
        # initial compiled pattern
        self.compiled_pattern = re.compile(self.pattern)
        # special token dictionary, {str: int}
        # ex: {'<|endoftext|>': 100257}
        self.special_tokens = {}
        # inverse above dictionary, {int: str}
        # ex: {100257: '<|endoftext|>'}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        # train a vocab base on input text
        # vocab size have to greater or equal than 256 since in original utf-8 encode will have 255 int token
        assert vocab_size >= 256
        # calculate how many times merge needed
        num_merges = vocab_size - 256

        # split the text up into text chunks, according to above pattern, ex: ['Copy', ' paste', ' of', ' the']
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing, list of each chunk list, ex: [[67, 111, 112, 121], [32, 112, 97, 115, 116, 101]]
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # iteratively merge the most common pairs to create new tokens
        # (int, int) -> int, ex: (101, 32): 256
        merges = {}
        # int -> bytes, map token index into bytes string
        vocab = {idx: bytes([idx]) for idx in range(256)}
        # loop through all merge times
        for i in range(num_merges):
            # count the number of times every consecutive pair appears
            stats = {}
            # each chunk will be one split text convert to bytes int ids
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)
            # find the pair with the highest count
            # ex: (1, 2): 2, pair is (1, 2) with the highest count 2
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            # this is chunk by chunk merge, so different chunk ids will not be merged, ids: list of each chunk list
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
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

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        # get inverse dictionary int -> str
        # ex: {100257: '<|endoftext|>'}
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        # given ids (list of integers), return Python string
        # append raw bytes string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                # append raw bytes string
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                # append special token raw bytes string
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")

        # join list of bytes string
        text_bytes = b"".join(part_bytes)
        # convert bytes string into string, if face the error, will be replaced by special character like '?'
        text = text_bytes.decode("utf-8", errors="replace")

        return text

    def _encode_chunk(self, text_bytes):
        # input is raw bytes text, output the token ids integers
        # below is almost same as basic BPE token encode method
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        # merge until no more pair shows up inside merges dictionary or new ids length last than 2 (a pair)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
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
                break  # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            # get new merged idx
            idx = self.merges[pair]
            # merge ids list into new ids according to new idx
            ids = merge(ids, pair, idx)

        return ids

    def encode_ordinary(self, text):
        """
        Encoding that ignores any special tokens.
        """
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        # encode chunk by chunk
        for chunk in text_chunks:
            # raw bytes encode
            chunk_bytes = chunk.encode("utf-8")
            # get chunk token ids
            chunk_ids = self._encode_chunk(chunk_bytes)
            # join into final output ids
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            # any special tokens should not show in text
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))

        return ids
