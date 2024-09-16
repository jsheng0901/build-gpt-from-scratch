import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    # super simple bi-gram model
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # embedding -> [65, 65], each vocab embedding to same size dimension
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        # B -> batch size, T -> time (block size), C -> channels embedding size (here also equal class number)
        # idx -> [32, 8] * embedding table -> [65, 65] = [32, 8, 65] * [65, 65] = [32, 8, 65]
        # [32， 8， 65] means transfer each word index to 65 length 0-1 vector,
        # only index position will be 1, others all 0
        logits = self.token_embedding_table(idx)

        # if target is None means forward on generate step
        if targets is None:
            loss = None
        else:
            # B -> 32, T -> 8, C -> 65
            B, T, C = logits.shape
            # [32， 8， 65] -> [32 * 8, 65] -> [256, 65]
            logits = logits.view(B * T, C)
            # [32, 8] -> [32 * 8, ] -> [256, ]
            targets = targets.view(B * T)
            # calculate each word raw predict probability without softmax on 65 classes loss
            # cross entropy loss = -mean(sum_sample(sum_class((y_true * log(y_pred)))))
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        # idx -> [1, 1], batch -> 1, length -> 1 length will increase with new generate idx concat
        for _ in range(max_new_tokens):
            # get the predictions
            # logits -> [1, 1, 65] raw predict value, length will increase same as idx
            logits, loss = self(idx)
            # focus only on the last time step, because we only care about last time step predict value
            # becomes [B, C] -> [1, 65], means each batch raw predict each class value
            logits = logits[:, -1, :]
            # apply softmax to get probabilities, [B, C] -> [1, 65]
            probs = F.softmax(logits, dim=-1)
            # get one sample index according to each class probability from multi nominal distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence, [B, T+1] -> [1, 1 + 1] -> [1, 2], this will be new input
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, block_size, dropout, head_size):
        super().__init__()
        # key linear layer with weight, ex: [384, 64]
        self.key = nn.Linear(n_embd, head_size, bias=False)
        # query linear layer with weight, ex: [384, 64]
        self.query = nn.Linear(n_embd, head_size, bias=False)
        # value linear layer with weight, ex: [384, 64]
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # create lower triangular part of the matrix, upper part all 0, this will be as mask in key value attention
        # ex: [time, time] -> [block_size, block_size] -> [256, 256]
        # register_buffer will create a matrix which will not be considered as parameters
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)

        # ex: B -> 64, T -> 256, C -> 386
        B, T, C = x.shape
        # [batch, time, embedding] * [batch, embedding, head_size] = [batch, time, head_size]
        # ex: [384, 64] -> x: [64, 256, 386] * k: [64, 386, 64] -> [64, 256, 64]
        k = self.key(x)
        # same as k, q -> [batch, time, head_size], ex: [64, 256, 64]
        q = self.query(x)
        # compute attention scores ("affinities")
        # q -> [batch, time, head_size]
        # k.transpose(-2, -1) -> [batch, head_size, time]
        # q @ k.transpose(-2, -1) -> [batch, time, head_size] @ [batch, head_size, time] -> [batch, time, time]
        # k.shape[-1] ** -0.5 -> sqrt(head_size) this is rescaled
        # ex: [64, 256, 64] * [64, 64, 256] -> [64, 256, 256]
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        # tril[:T, :T] == 0 -> [256, 256] with boolean equal to 0, here upper part all true, lower part all false
        # masked_fill will fill all true position with '-inf', this will help softmax to get 0, which means no attention
        # [batch, time, time] -> ex: [64, 256, 256]
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # [batch, time, time] -> ex: [64, 256, 256], softmax on last dimension
        wei = F.softmax(wei, dim=-1)
        # apply dropout layer
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        # same as k, v -> [batch, time, head_size], ex: [64, 256, 64]
        v = self.value(x)
        # [batch, time, time] * [batch, time, head_size] -> [batch, time, head_size]
        # ex: [64, 256, 256] * [64, 256, 64] -> [64, 256, 64]
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_embd, block_size, head_size, num_heads, dropout):
        super().__init__()
        # create multi heads according to above head class
        self.heads = nn.ModuleList([Head(n_embd, block_size, dropout, head_size) for _ in range(num_heads)])
        # linear project layer, project all heads output into embedding dimension
        # ex: [64 * 6, 384]
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input x -> [batch, time, n_embd]
        # h(x) -> [batch, time, head_size]
        # build each head and concat
        # [batch, time, head_size] * num_heads concat along head_size dimension
        # [64, 256, 64] -> [64, 256, 64 * 6] -> [64, 256, 384]
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # [batch, time, head_size * num_heads] @ [head_size * num_heads, n_embd] -> [batch, time, n_embd]
        # ex: [64, 256, 384] * [64 * 6, 384] -> [64, 256, 384]
        # output dimension will be same like input x -> [batch, time, n_embd]
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, dropout, n_embd):
        super().__init__()
        # FFN layer stack
        self.net = nn.Sequential(
            # linear projection layer [n_embd, 4 * n_embd], ex: [384, 384 * 4]
            nn.Linear(n_embd, 4 * n_embd),
            # activate layer, in original code is gelu
            nn.ReLU(),
            # linear projection back layer [4 * n_embd, n_embd], ex: [384 * 4, 384]
            nn.Linear(4 * n_embd, n_embd),
            # dropout layer
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # input will be multi head attention output
        # x -> [batch, time, n_embd]
        # ex: [64, 256, 384] * [384, 384 * 4] * [384 * 4, 384] -> [64, 256, 384]
        # output dimension will be same like input x -> [batch, time, n_embd]
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, block_size, num_heads, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        # head size will be even distribute according to number of heads
        # ex: 384 // 6 = 64
        head_size = n_embd // num_heads
        # input -> [batch, time, n_embd] output -> [batch, time, n_embd]
        self.sa = MultiHeadAttention(n_embd, block_size, head_size, num_heads, dropout)
        # input -> [batch, time, n_embd] output -> [batch, time, n_embd]
        self.ffwd = FeedFoward(dropout, n_embd)
        # LN layer before attention
        self.ln1 = nn.LayerNorm(n_embd)
        # LN layer before FFN
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # intput x -> [batch, time, n_embd]
        # output x -> [batch, time, n_embd]
        # with residual connection
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size, block_size, n_embd, n_layer, num_heads, dropout, device):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # embedding -> [65, 384]
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # position embedding -> [256, 384]
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # stack each block, here we have 6 layers, block input output same -> [batch, time, n_embd]
        self.blocks = nn.Sequential(*[Block(n_embd, block_size, num_heads, dropout) for _ in range(n_layer)])
        # final layer norm
        self.ln_f = nn.LayerNorm(n_embd)
        # linear layer project to output class -> [n_embd, vocab_size], ex: [384, 65]
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, apply init weights function recursively on every submodule
        self.apply(self._init_weights)

        # save parameters
        self.device = device
        self.block_size = block_size

    def _init_weights(self, module):
        # initial parameters weights, this is important as will influence converge speed and accuracy
        # linear layer initial
        if isinstance(module, nn.Linear):
            # fill the input Tensor with values drawn from the normal distribution.
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # if linear layer has bias, then fill the input Tensor with the scalar value 0
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # embedding layer initial
        elif isinstance(module, nn.Embedding):
            # same as linear layer initial
            # fill the input Tensor with values drawn from the normal distribution.
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # B -> batch, T -> time (block size)
        # ex: B -> 64, T -> 256
        B, T = idx.shape

        # idx and targets are both [B, T] tensor of integers
        # B -> batch size, T -> time (block size), C -> channels embedding size
        # idx -> [64, 256] * embedding table -> [65, 384] = [64, 256, 65] * [65, 384] = [64, 256, 384]
        # [64, 256, 65] means transfer each word index to 65 length 0-1 vector,
        # only index position will be 1, others all 0
        tok_emb = self.token_embedding_table(idx)
        # position vector is [0, 1, 2, ..., T], ex: [0, 1, 2, ... 255]
        # [1, 256] -> [256, 256] * [256, 384] -> [256, 384], same as token embedding logic, pos_emb -> [time, embedding]
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        # tok_emb -> [batch, time, embedding] + pos_emb -> [time, embedding] = [batch, time, embedding]
        x = tok_emb + pos_emb
        # input x -> [batch, time, n_embd], output x -> [batch, time, n_embd]
        x = self.blocks(x)
        # LN layer for input x -> [batch, time, n_embd]
        x = self.ln_f(x)
        # [batch, time, n_embd] * [n_embd, vocab_size] -> [batch, time, vocab_size]
        # ex: [64, 256, 384] * [384, 65] -> [64, 256, 65]
        logits = self.lm_head(x)

        # if target is None means forward on generate step
        if targets is None:
            loss = None
        else:
            # B -> 64, T -> 256, C -> 65
            B, T, C = logits.shape
            # ex: [64, 256, 65] -> [64 * 256, 65]
            logits = logits.view(B * T, C)
            # [64, 256] -> [64 * 256, ]
            targets = targets.view(B * T)
            # calculate each word raw predict probability without softmax on 65 classes loss
            # cross entropy loss = -mean(sum_sample(sum_class((y_true * log(y_pred)))))
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        # idx -> [1, 1], batch -> 1, length -> 1 length will increase with new generate idx concat
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens, since max_new_tokens may longer than input block size (time)
            # ex: max_new_tokens -> 500, idx -> [64, 400], then always use last 256 length tokens, idx -> [64, 256]
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            # ex: logits -> [1, 1, 65] raw predict value, length will increase same as idx
            logits, loss = self(idx_cond)
            # focus only on the last time step, because we only care about last time step predict value
            # ex: becomes [B, C] -> [1, 65], means each batch raw predict each class value
            logits = logits[:, -1, :]
            # apply softmax to get probabilities, [B, C] -> [1, 65]
            probs = F.softmax(logits, dim=-1)
            # get one sample index according to each class probability from multi nominal distribution
            # ex: [B, 1] -> [1, 1]
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence, [B, T+1] -> [1, 1 + 1] -> [1, 2], this will be new input
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
