import torch
from Transformer import Transformer


if __name__ == "__main__":
    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(
        device
    )
    # 定义了源序列和目标序列的padding索引。在这个例子中，它们都被设为0，意味着索引0将被用于填充序列使它们长度一致
    src_pad_idx = 0
    trg_pad_idx = 0
    # 定义了源和目标词汇的大小，都被设置为10。这通常代表模型能识别的不同符号的数量
    src_vocab_size = 10
    trg_vocab_size = 10

    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(
        device
    )
    out = model(x, trg[:, :-1])
    print(out.shape)