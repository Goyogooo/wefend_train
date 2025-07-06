import torch
import torch.nn as nn
import torch.nn.functional as F

class AggregationCell(nn.Module):
    """
    聚合单元
    """
    def __init__(self, output_dim=20,dropout=0.3):
        super(AggregationCell, self).__init__()
        self.fc = nn.Linear(in_features=40, out_features=output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, report_features, lengths=None, training=False):
        """
        report_features: Tensor(shape=[total_reports, feat_dim])，拼接后的所有报告特征
        lengths: Tensor(shape=[batch])，每个样本包含的报告数
        返回:
          Tensor(shape=[batch, output_dim])
        """
     
        # RaggedTensor：先 sum，然后除以每行的长度
        # 拆分张量，每个元素是单个样本的举报特征
        split_features = torch.split(report_features, lengths.tolist(), dim=0)
        # sum: (batch, feat_dim)聚合每个样本的特征和
        summed = torch.stack([x.sum(dim=0) for x in split_features], dim=0)
        lengths = lengths.unsqueeze(1).to(dtype=report_features.dtype)
        pooled = summed / lengths  # (batch, feat_dim)

        # 先 fc + ReLU，再 Dropout
        out = F.relu(self.fc(pooled))
        out = self.dropout(out)
        return out    # (batch, output_dim)
