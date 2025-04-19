import torch
from torch import nn

'''
 3.3 Position-wise Feed-Forward Networks
 In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully
 connected feed-forward network, which is applied to each position separately and identically. This
 consists of two linear transformations with a ReLU activation in between.
 FFN(x) = max(0xW1 +b1)W2 +b2
 (2)
 While the linear transformations are the same across different positions, they use different parameters
 from layer to layer. Another way of describing this is as two convolutions with kernel size 1.
 The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality
 dff = 2048.
'''


class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super().__init__()
        self.ln1 = nn.Linear(d_model, hidden)
        self.ln2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.relu(self.ln1(x))
        x = self.dropout(x)
        x = self.ln2(x)
        return x
