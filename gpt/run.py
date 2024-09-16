import torch
import logging
from train import BigramLanguageModelTrainer, GPTLanguageModelTrainer
from config import BigramLanguageModelConfig, GPTLanguageModelConfig

# set up logging root level
logging.getLogger().setLevel(logging.INFO)


def bigram_language_model_run(context):
    # load the config
    config = BigramLanguageModelConfig().all_config

    # train the model
    trainer = BigramLanguageModelTrainer(config)
    trainer.train()

    # generate from the model
    context = context.to(config['device'])
    idx_output = trainer.model.generate(idx=context, max_new_tokens=config['max_new_tokens'])[0].tolist()
    string_output = trainer.dataloader.decode(idx_output)
    logging.info(f"generate new sentence: {string_output}")


def gpt_language_model_run(context):
    # load the config
    config = GPTLanguageModelConfig().all_config

    # train the model
    trainer = GPTLanguageModelTrainer(config)
    trainer.train()

    # generate from the model
    context = context.to(config['device'])
    idx_output = trainer.model.generate(idx=context, max_new_tokens=config['max_new_tokens'])[0].tolist()
    string_output = trainer.dataloader.decode(idx_output)
    logging.info(f"generate new sentence: {string_output}")


# start index with 0, which is '\n' in this dataset vocab
context = torch.zeros((1, 1), dtype=torch.long)
gpt_language_model_run(context)
