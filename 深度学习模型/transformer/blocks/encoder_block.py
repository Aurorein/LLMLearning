import torch.nn as nn
from torch.nn import LayerNorm

from layers.multi_head_attention import MultiHeadAttention
from layers.position_wise_feed_forward import PositionWiseFeedForward

'''
Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two
 sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position
wise fully connected feed-forward network. We employ a residual connection [11] around each of
 the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is
 LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer
 itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding
 layers, produce outputs of dimension dmodel = 512
'''


class EncoderLayer(nn.Module):
    def __init__(self, ffn_hidden, n_head, drop_prob, d_model=512):
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        # 归一化
        self.norm1 = LayerNorm(d_model, eps=1e-12)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionWiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model, eps=1e-12)
        self.dropout2 = nn.Dropout(p=drop_prob)

    # N = 6
    def forward(self, x, src_mask):
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        x = self.dropout1(x)
        # 残差网络 residual connections
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)

        x = self.dropout2(x)
        x = self.norm2(x + _x)

        return x
