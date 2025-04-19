import torch.nn as nn
import embedding.token_embedding as TokenEmbedding
import embedding.positional_encoding as PositionEncoding

class TransformerEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model, max_len, drop_prob = 0.1, device):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_ecd = PositionEncoding(d_model, max_len, device)
        self.dropout = nn.Dropout(p = drop_prob)

    def forward(self, x):
        tok_emb_x = self.tok_emb(x)
        pos_ecd_x = self.pos_ecd(x)
        return self.dropout(tok_emb_x + pos_ecd_x)