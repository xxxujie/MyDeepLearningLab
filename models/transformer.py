import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_count):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_count = head_count
        self.q_weight = nn.Linear(embedding_dim, embedding_dim)
        self.k_weight = nn.Linear(embedding_dim, embedding_dim)
        self.v_weight = nn.Linear(embedding_dim, embedding_dim)
        # 输出权重矩阵W_O
        self.output_weight = nn.Linear(embedding_dim, embedding_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q_seq, k_seq, v_seq):
        queries: Tensor = self.q_weight(q_seq)
        keys: Tensor = self.k_weight(k_seq)
        values: Tensor = self.v_weight(v_seq)
        # 拆分多头
        batch_size, seq_len, embedding_dim = q_seq.shape
        head_dim = self.embedding_dim // self.head_count
        # 即最后一维拆分 -> embedding_dim = head_count * head_dim，并交换head_count和seq_dim维度
        queries = (
            queries.contiguous()
            .view(batch_size, seq_len, self.head_count, head_dim)
            .permute(0, 2, 1, 3)
        )
        keys = (
            keys.contiguous()
            .view(batch_size, seq_len, self.head_count, head_dim)
            .permute(0, 2, 1, 3)
        )
        values = (
            values.contiguous()
            .view(batch_size, seq_len, self.head_count, head_dim)
            .permute(0, 2, 1, 3)
        )
        # 计算注意力
        # 先获取一个mask，它是一个下三角矩阵
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=bool))
        attention_scores, _ = _attention(queries, keys, values, mask)
        # 合并多头
        attention_scores = (
            attention_scores.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, embedding_dim)
        )
        output = self.output_weight(attention_scores)
        return output


def _attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
    dropout: nn.Dropout = None,
) -> tuple[Tensor, Tensor]:
    d_k = query.shape(-1)
    # torch的矩阵乘法支持带batch的乘法，因此二维以上的矩阵也可以相乘
    score_probs = query @ key.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        # mask == 0的位置都设置为负无穷
        score_probs = score_probs.masked_fill(mask == 0, float("-inf"))
    score_probs = F.softmax(score_probs, dim=-1)
    if dropout is not None:
        score_probs = dropout(score_probs)
    return score_probs @ value, score_probs
