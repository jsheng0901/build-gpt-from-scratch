"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds.
"""

import os
import time
from basic_tokenizer import BasicTokenizer
from regex_tokenizer import RegexTokenizer


def train_tokenizer(text, vocab_size):
    print(f"Training vocab size is {vocab_size}")
    t0 = time.time()
    for TokenizerClass, name in zip([BasicTokenizer, RegexTokenizer], ["basic", "regex"]):

        # construct the Tokenizer object and kick off verbose training
        print(f"------------------------------------------------------")
        print(f"Start training {name} tokenizer")
        tokenizer = TokenizerClass()
        tokenizer.train(text, vocab_size, verbose=True)
        # writes two files in the models directory: name.model, and name.vocab
        prefix = os.path.join("models", name)
        tokenizer.save(prefix)
    t1 = time.time()

    print(f"Training took {t1 - t0:.2f} seconds")


# open some text and train a vocab of 512 tokens
text = open("tests/taylorswift.txt", "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

train_tokenizer(text, 4096)
