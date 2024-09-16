from dataclasses import dataclass

data_config = {
    "dataset_path": "input.txt",
    "block_size": 256,
    "batch_size": 64
}

model_config = {
    "max_new_tokens": 500,
    "learning_rate": 3e-4,
    "num_epochs": 3000,
    "eval_interval": 500,
    "eval_iters": 200,
    "device": "mps",
    "n_embd": 384,
    "num_heads": 6,
    "n_layer": 6,
    "dropout": 0.2
}


@dataclass
class BigramLanguageModelConfig:
    all_config = {
        **data_config,
        **model_config,
    }


class GPTLanguageModelConfig:
    all_config = {
        **data_config,
        **model_config,
    }
