import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=500):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, embed_dim, heads):

        super().__init__()

        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads

        assert (
            self.head_dim * heads == embed_dim
        ), "Embed size needs divisible by heads"

        self.values = nn.Linear(embed_dim, embed_dim)
        self.keys = nn.Linear(embed_dim, embed_dim)
        self.queries = nn.Linear(embed_dim, embed_dim)

        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        N = x.shape[0]
        seq_len = x.shape[1]

        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        values = values.reshape(
            N,
            seq_len,
            self.heads,
            self.head_dim
        )

        keys = keys.reshape(
            N,
            seq_len,
            self.heads,
            self.head_dim
        )

        queries = queries.reshape(
            N,
            seq_len,
            self.heads,
            self.head_dim
        )

        energy = torch.einsum(
            "nqhd,nkhd->nhqk",
            queries,
            keys
        )

        attention = torch.softmax(
            energy / (self.head_dim ** 0.5),
            dim=-1
        )

        out = torch.einsum(
            "nhql,nlhd->nqhd",
            attention,
            values
        )

        out = out.reshape(
            N,
            seq_len,
            self.embed_dim
        )

        out = self.fc_out(out)

        return out, attention


class TransformerModel(nn.Module):

    def __init__(
        self,
        num_items,
        embed_dim=32,
        heads=2
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            num_items,
            embed_dim
        )

        self.positional_encoding = PositionalEncoding(
            embed_dim
        )

        self.attention = MultiHeadSelfAttention(
            embed_dim,
            heads
        )

        self.dropout = nn.Dropout(0.2)

        self.norm = nn.LayerNorm(embed_dim)

        self.fc = nn.Linear(
            embed_dim,
            num_items
        )

    def forward(self, x):

        x = self.embedding(x)

        x = self.positional_encoding(x)

        attn_out, attention = self.attention(x)

        x = self.norm(x + attn_out)

        x = self.dropout(x)

        x = x[:, -1, :]

        logits = self.fc(x)

        return logits, attention