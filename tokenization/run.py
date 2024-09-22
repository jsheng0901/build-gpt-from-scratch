import os

import tiktoken

from basic_tokenizer import BasicTokenizer
from tokenization.gpt4_tokenizer import GPT4Tokenizer
from tokenization.regex_tokenizer import RegexTokenizer


def run_inference(text):
    for TokenizerClass, name in zip([BasicTokenizer, RegexTokenizer, GPT4Tokenizer], ["basic", "regex", "GPT4"]):

        # construct the Tokenizer object and kick off encode and decode
        print(f"------------------------------------------------------")
        print(f"Start load {name} tokenizer")
        # load name.model files in the models directory
        if name != "GPT4":
            prefix = os.path.join("models", f"{name}.model")
            tokenizer = TokenizerClass()
            tokenizer.load(prefix)
        else:
            tokenizer = TokenizerClass()

        encode_token = tokenizer.encode(text)
        decode_text = tokenizer.decode(encode_token)
        print(f"This is encode text: {text}")
        print(f"This is encode token: {encode_token}")
        print(f"This is decode text: {decode_text}")
        print(f"The text after encode and decode is same: {decode_text == text}")


def test_gpt4_tiktoken_equality(text):
    # tiktoken tokenizer
    enc = tiktoken.get_encoding("cl100k_base")
    tiktoken_ids = enc.encode(text)
    print(f"This is tiktoken ids: {tiktoken_ids}")
    # GPT4 tokenizer
    tokenizer = GPT4Tokenizer()
    gpt4_tokenizer_ids = tokenizer.encode(text)
    print(f"This is out GPT4 tokenizer ids: {gpt4_tokenizer_ids}")
    print(f"The text after both encode are same: {gpt4_tokenizer_ids == tiktoken_ids}")


if __name__ == "__main__":
    # encode the text
    text = "hello world!!!? (안녕하세요!) lol123 😉"
    print("This is inference step")
    run_inference(text)
    print(f"------------------------------------------------------")
    print("This is test step")
    test_gpt4_tiktoken_equality(text)
