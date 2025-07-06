import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    """
    词向量层：将词 ID 映射到密集向量，即加载 DSG 预训练权重
    """
    def __init__(self, vocab_size, embedding_dim, embedding_matrix=None, trainable=False):
        """
        词向量层
        参数：
          vocab_size: 词表大小
          embedding_dim: 词向量维度
          embedding_matrix: 预训练的词向量矩阵
          trainable: 是否允许微调
        """
        super().__init__()
        if embedding_matrix is not None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)# 创建一个 nn.Embedding 层
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32),
                                                 requires_grad=trainable)# 用传入的 embedding_matrix 初始化 weight 参数
        else:
            # 随机初始化词向量
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            nn.init.uniform_(self.embedding.weight, a=-0.05, b=0.05) 
            self.embedding.weight.requires_grad = True

    def forward(self, inputs):
        """
        inputs: 整数 ID 张量，shape=(batch_size, seq_length)
        return: 词向量张量，shape=(batch_size, seq_length, embedding_dim)
        """
        return self.embedding(inputs)
