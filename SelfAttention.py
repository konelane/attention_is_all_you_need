import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """ 
    这个SelfAttention模块可以捕捉输入序列中不同位置之间的关系，
    这是通过计算每个位置对所有其他位置的注意力分数来实现的。
    这些分数表明了在生成每个位置的输出时应该给予序列中其他位置多少注意力。

    """

    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # 我们同时发送的example数量
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.head_dim)
        keys = keys.reshape(N, key_len, self.head_dim)
        queries = query.reshape(N, query_len, self.head_dim)

        # reshape后要发送会线性层
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)

        # 我们应该对input的每个单词
        # 
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # after einsum: (N, query_len, heads, heads_dim) then flatten last two dims

        # 通过fc_out发送出去
        out = self.fc_out(out)
        return out
