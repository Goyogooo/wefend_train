# Q
import torch
import torch.nn as nn
import torch.nn.functional as F
from embedding_layer import EmbeddingLayer 

class TextCNNEncoder(nn.Module):
    """
    TextCNN 编码器
    ————
    将一条举报文本编码成 40 维特征向量
    结构：
      1. EmbeddingLayer → (batch, seq_len, emb_dim)
      2. expand_dims → (batch, seq_len, emb_dim, 1)
      3. 多个 Conv2D+ReLU → (batch, seq_len-fs+1, 1, num_filters)
      4. 时间维度 max-pool → (batch, num_filters)
      5. 拼接所有 filter 输出 → (batch, num_filters*6=240)
      6. Dropout
      7. Dense(40)+ReLU → (batch, 40)
    """
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 embedding_matrix,
                 filter_sizes=[1, 2, 3, 4, 5, 6],
                 num_filters=40,
                 feature_dim=40,
                 dropout_rate=0.5,
                 embedding_trainable=False):
        super(TextCNNEncoder, self).__init__()
        # 词嵌入层
        self.embedding = EmbeddingLayer(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            embedding_matrix=embedding_matrix,
            trainable=embedding_trainable
        )
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        # 定义多个卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=num_filters,
                kernel_size=(fs, embedding_dim)
            )
            for fs in filter_sizes
        ])
        
        # Dropout 防止过拟合
        self.dropout = nn.Dropout(dropout_rate)
        # 将 240 维特征映射到 40 维
        self.projection = nn.Linear(num_filters * len(filter_sizes), feature_dim)

    def forward(self, inputs, training=False):
        """
        inputs: (batch_size, seq_length)
        return: (batch_size, feature_dim=40)
        """
        # 1. 词嵌入
        x = self.embedding(inputs)                     # (batch, seq_len, emb_dim)
        # 2. 增加通道维度
        x = x.unsqueeze(-1)                            # (batch, seq_len, emb_dim, 1)
        x = x.permute(0, 3, 1, 2)                      # → (batch, 1, seq_len, emb_dim)

        pooled_outputs = []
        # 对每个 filter size 做卷积+池化
        for conv in self.convs:
            c = F.relu(conv(x))                         # (batch, num_filters, seq_len-fs+1, 1)
            p = torch.max(c, dim=2).values              # (batch, num_filters, 1) 在时间维度上做 max-pool
            p = p.squeeze(2)                            # (batch, num_filters)
            pooled_outputs.append(p)

        # 拼接所有 filter 输出
        h = torch.cat(pooled_outputs, dim=-1)           # (batch, num_filters*6 = 240)
        # Dropout
        if self.training:
            h = self.dropout(h)        # (batch, 240)
        # 全连接降维到 40
        h = F.relu(self.projection(h))                  # (batch, 40)
        return h


# 我们使用 40 个窗口大小从 1 到 6 不等的过滤器，完全连接层将特征的维数从 240 调整到 40。
# 最后一个全连接层以 40 维特征向量作为输入来识别新闻是否为假新闻。
