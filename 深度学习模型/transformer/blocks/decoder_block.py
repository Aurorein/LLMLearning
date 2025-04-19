import torch
import torch.nn as nn
from torch.nn import LayerNorm

from layers.multi_head_attention import MultiHeadAttention
from layers.position_wise_feed_forward import PositionWiseFeedForward

'''
Decoder: Thedecoder is also composed of a stack of N = 6 identical layers. In addition to the two
 sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head
 attention over the output of the encoder stack. Similar to the encoder, we employ residual connections
 around each of the sub-layers, followed by layer normalization. We also modify the self-attention
 sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This
 masking, combined with fact that the output embeddings are offset by one position, ensures that the
 predictions for position i can depend only on the known outputs at positions less than i.
'''


class DecoderLayer(nn.Module):
    def __init__(self, ffn_hidden, n_head, drop_prob, d_model=512):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionWiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, enc, dec, trg_mask, src_mask):

        # self-attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        # add & norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            _x = enc
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x