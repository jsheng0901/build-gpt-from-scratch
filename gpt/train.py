import logging
import torch

from data_load import DataLoad
from model import BigramLanguageModel, GPTLanguageModel


class BigramLanguageModelTrainer:
    def __init__(self, config):
        # data parameters
        self.dataset_path = config['dataset_path']
        self.block_size = config['block_size']
        self.batch_size = config['batch_size']

        # model parameters
        self.learning_rate = config['learning_rate']
        self.num_epochs = config['num_epochs']
        self.device = config['device']
        self.eval_interval = config['eval_interval']
        self.eval_iters = config['eval_iters']

        # preprocess data
        self.dataloader = DataLoad(self.dataset_path, self.batch_size, self.block_size)
        self.dataloader.create_train_val()

        # create the model
        self.model = BigramLanguageModel(self.dataloader.vocab_size).to(self.device)

    def train(self):
        # create optimizer and learning rate scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # train and eval
        for epoch in range(self.num_epochs):
            # start train mode
            self.model.train()
            # random generate one batch
            # xb -> [32, 8] yb -> [32, 8]
            xb, yb = self.dataloader.get_batch('train')

            # evaluate the loss
            # logits -> [256, 65] raw predict value, loss -> single float value
            logits, loss = self.model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # every once in a while evaluate the loss on train and val sets
            if epoch % self.eval_interval == 0:
                self.model.eval()
                out = {}
                for split in ['train', 'val']:
                    # evaluate iters means how many batch data will be generated to run evaluate
                    losses = torch.zeros(self.eval_iters)
                    for k in range(self.eval_iters):
                        x_eval, y_eval = self.dataloader.get_batch(split)
                        with torch.no_grad():
                            logits, loss = self.model(x_eval, y_eval)
                        # store each iter loss
                        losses[k] = loss.item()
                    # store each dataset mean loss
                    out[split] = losses.mean()

                logging.info(f"step {epoch}: train loss {out['train']:.4f}, val loss {out['val']:.4f}")

        # save model in local
        # self.model.save(self.model_id)

        return


class GPTLanguageModelTrainer:
    def __init__(self, config):
        # data parameters
        self.dataset_path = config['dataset_path']
        self.block_size = config['block_size']
        self.batch_size = config['batch_size']

        # model parameters
        self.n_embd = config['n_embd']
        self.n_layer = config['n_layer']
        self.num_heads = config['num_heads']
        self.dropout = config['dropout']
        self.learning_rate = config['learning_rate']
        self.num_epochs = config['num_epochs']
        self.device = config['device']
        self.eval_interval = config['eval_interval']
        self.eval_iters = config['eval_iters']

        # preprocess data
        self.dataloader = DataLoad(self.dataset_path, self.batch_size, self.block_size, self.device)
        self.dataloader.create_train_val()

        # create the model
        self.model = GPTLanguageModel(self.dataloader.vocab_size, self.block_size, self.n_embd, self.n_layer,
                                      self.num_heads, self.dropout, self.device).to(self.device)

        # print the number of parameters in the model
        # torch.numel() will get total number of elements in the input tensor.
        # then recursively calculate all parameters total elements and sum together
        logging.info(f"{sum(p.numel() for p in self.model.parameters()) / 1e6} M parameters")

    def train(self):
        # set random seed, only need set one time in here, don't set in dataloader,
        # otherwise will cause each time generate batch, data distribution too random, model training converge hard
        torch.manual_seed(1337)
        # create optimizer and learning rate scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # train and eval
        for epoch in range(self.num_epochs):
            # start train mode
            self.model.train()
            # random generate one batch
            # xb -> [64, 256] yb -> [64, 256]
            xb, yb = self.dataloader.get_batch('train')

            # evaluate the loss
            # logits -> [64 * 256, 65] raw predict value, loss -> single float value
            logits, loss = self.model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # logging.info(f"step {epoch}: train loss {loss}")

            # every once in a while evaluate the loss on train and val sets
            if epoch % self.eval_interval == 0 or epoch == self.num_epochs - 1:
                self.model.eval()
                out = {}
                for split in ['train', 'val']:
                    # evaluate iters means how many batch data will be generated to run evaluate
                    losses = torch.zeros(self.eval_iters)
                    for k in range(self.eval_iters):
                        x_eval, y_eval = self.dataloader.get_batch(split)
                        with torch.no_grad():
                            logits, loss = self.model(x_eval, y_eval)
                        # store each iter loss
                        losses[k] = loss.item()
                    # store each dataset mean loss
                    out[split] = losses.mean()

                logging.info(f"step {epoch}: eval train loss {out['train']:.4f}, val loss {out['val']:.4f}")

        # save model in local
        # self.model.save(self.model_id)

        return
