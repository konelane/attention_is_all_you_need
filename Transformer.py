import torch
import torch.nn as nn

from SelfAttention import SelfAttention
from Encoder import Encoder
from Decoder import Decoder, DecoderBlock


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embed_size=256,
                 num_layers=6,
                 forward_expansion=4,
                 heads=8,
                 dropout=0,
                 device="cuda",
                 max_length=100

                 ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        # 设置源序列和目标序列的填充索引
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # 定义了一个函数来创建源数据的掩码，这个掩码会用于在自注意力计算中遮蔽填充token。
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(1)
        # (N, 1, 1, src_len)
        # src_mask 被设计成一个四维张量，其中两个新添加的维度大小为1。
        # 这样的形状是为了让它能够和Transformer中使用的多头注意力机制兼容，因为在多头注意力中通常需要三维或四维的张量。
        # 具体来说，src_mask 能够与源序列的批处理大小、序列长度以及多头注意力机制的维度相匹配。
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        # 下三角矩阵 值是1
        # 创建一个下三角矩阵（即未来token位置为0，其他位置为1），然后扩展到适合批次大小的四维矩阵。
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)

        return out
