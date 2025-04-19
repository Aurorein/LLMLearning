import torch.nn as nn


class TokenEmbedding(nn.Embedding):

    def __init__(self, vocab_size, vocab_dim):
        super().__init__(vocab_size, vocab_dim, padding_idx=1)
