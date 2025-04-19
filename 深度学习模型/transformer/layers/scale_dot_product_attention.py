import math

import torch
import torch.nn as nn

'''
We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of
 queries and keys of dimension dk, and values of dimension dv. We compute the dot products of the
 query with all keys, divide each by dk, and apply a softmax function to obtain the weights on the
 values.
 In practice, we compute the attention function on a set of queries simultaneously, packed together
 into a matrix Q. The keys and values are also packed together into matrices K and V . We compute
 the matrix of outputs as:
 Attention(QKV ) = softmax(QKT
 dk 
)V
 (1)
 The two most commonly used attention functions are additive attention [2], and dot-product (multi
plicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor
 of 1
 dk 
. Additive attention computes the compatibility function using a feed-forward network with
 a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is
 much faster and more space-efficient in practice, since it can be implemented using highly optimized
 matrix multiplication code.
 While for small values of dk the two mechanisms perform similarly, additive attention outperforms
 dot product attention without scaling for larger values of dk [3]. We suspect that for large values of
 dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has
 extremely small gradients 4. To counteract this effect, we scale the dot products by 1
 dk 
.
'''


class ScaleDotProductAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax()

    def forward(self, q, k, v, mask=None):
        # k [batch_size, n_head, n_len, d_tensor]
        batch_size, n_head, n_len, d_tensor = k.size()
        k_t = k.transpose(2, 3)
        # k_t [batch_size, n_head, d_tensor, n_len]
        score = q @ k / math.sqrt(d_tensor)
        # score [batch_size, n_head, n_len, n_len]
        # 去掉维度的影响，防止维度过大
        if mask is not None:
            score = score.mask_fill(mask == 0, -10000)
        # 掩码为0的地方填上-10000
        score = self.softmax(score)
        # score进行归一化
        v = score @ v
        # (n_len, n_len) @ (n_len, d_tensor)
        # v [batch, n_head, n_len, d_tensor]
        return v, score
