import torch.nn as nn
import torch

'''
The positional encodings have the same dimension dmodel
 as the embeddings, so that the two can be summed. There are many choices of positional encodings,
 learned and fixed [9].
 In this work, we use sine and cosine functions of different frequencies:
 PE(pos2i) = sin(pos 100002i dmodel)
 PE(pos2i+1) = cos(pos 100002i dmodel)
 where pos is the position and i is the dimension. That is, each dimension of the positional encoding
 corresponds to a sinusoid. The wavelengths form a geometric progression from 2 to 10000 2 . We
 chose this function because we hypothesized it would allow the model to easily learn to attend by
 relative positions, since for any fixed offset k, PEpos+k can be represented as a linear function of
 PEpos.
 Wealso experimented with using learned positional embeddings [9] instead, and found that the two
 versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version
 because it may allow the model to extrapolate to sequence lengths longer than the ones encountered
 during training.
'''
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len, device):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device = device).float()
        pos = pos.unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model))).float()
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model))).float()

    def forward(self, x):
        batch_size, seq_len = x.size()

        return self.encoding[:seq_len, :]