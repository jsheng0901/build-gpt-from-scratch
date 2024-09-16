import torch


class DataLoad:
    # this tokenization is very simple character level token.
    def __init__(self, path, batch_size, block_size, device):
        # read it in to inspect it
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        # here are all the unique characters that occur in this text
        chars = sorted(list(set(text)))

        # create a mapping from characters to integers
        # here we use character level vocab
        self.s_to_i = {ch: i for i, ch in enumerate(chars)}
        self.i_to_s = {i: ch for i, ch in enumerate(chars)}

        # add class field
        # how many independent sequences will we process in parallel
        self.block_size = block_size
        # what is the maximum context length for predictions?
        self.batch_size = batch_size
        # vocab size
        self.vocab_size = len(chars)
        # device where put data
        self.device = device
        # whole data text
        self.text = text
        # train dataset
        self.train_data = None
        # val dataset
        self.val_data = None

    def encode(self, string):
        # encoder: take a string, output a list of integers
        res = [self.s_to_i[c] for c in string]
        return res

    def decode(self, list_integers):
        # decoder: take a list of integers, output a string
        string = ''.join([self.i_to_s[i] for i in list_integers])
        return string

    def create_train_val(self):
        # encode the entire text dataset and store it into a torch.Tensor
        data = torch.tensor(self.encode(self.text), dtype=torch.long)

        # split up the data into train and validation sets
        # first 90% will be trained, rest val
        # train dataset size
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == 'train' else self.val_data
        # random generate batch size start sentence index, here it's [4, ]
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        # create sentence according to above generate start index, x -> [4, 8]
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        # create sentence label by shift right one index, y -> [4, 8]
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y


# test
# dataloader = DataLoad('input.txt', 4, 8, 'mps')
# dataloader.create_train_val()
# xb, yb = dataloader.get_batch('train')
# print('inputs:')
# print(xb.shape)
# print(xb)
# print('targets:')
# print(yb.shape)
# print(yb)
#
# print('----')
#
# for b in range(dataloader.batch_size):  # batch dimension
#     for t in range(dataloader.block_size):  # time dimension
#         context = xb[b, :t + 1]
#         target = yb[b, t]
#         print(f"when input is {context.tolist()} the target: {target}")
