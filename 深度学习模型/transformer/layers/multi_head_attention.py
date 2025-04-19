import torch
import torch.nn as nn

from layers.scale_dot_product_attention import ScaleDotProductAttention

'''
 Instead of performing a single attention function with dmodel-dimensional keys, values and queries,
 we found it beneficial to linearly project the queries, keys and values h times with different, learned
 linear projections to dk, dk and dv dimensions, respectively. On each of these projected versions of
 queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional
 output values. These are concatenated and once again projected, resulting in the final values, as
 depicted in Figure 2.
 Multi-head attention allows the model to jointly attend to information from different representation
 subspaces at different positions. With a single attention head, averaging inhibits this.
 MultiHead(QKV) = Concat(head1
 headh)WO
 where headi = Attention(QWQ
 i KWK
 i VWV
 i )
 Where the projections are parameter matrices WQ
 i 
and WO Rhdv dmodel.
 Rdmodel dk, WK
 i
 Rdmodel dk, WV
 i 
Rdmodel dv
 In this work we employ h = 8 parallel attention layers, or heads. For each of these we use
 dk = dv =dmodel h = 64. Due to the reduced dimension of each head, the total computational cost
 is similar to that of single-head attention with full dimensionality
'''

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        # 进行线性变换

        q, k, v = self.split(q), self.split(k), self.split(v)
        out, score = self.attention.forward(q, k, v, mask=mask)
        out = self.concat(out)
        # 恢复特征空间的可学习性
        out = self.w_concat(out)

    def split(self, tensor):
        batch_size, n_len, d_model = tensor.size()
        n_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, n_len, self.n_head, n_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        batch_size, n_head, n_len, d_tensor = tensor.size()
        tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor