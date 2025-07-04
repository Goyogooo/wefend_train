import torch
import torch.nn as nn
import torch.nn.functional as F

class Annotator(nn.Module):
    """
    Annotator 模型
    """
    def __init__(self, text_cnn_encoder, aggregation_cell):
        super(Annotator, self).__init__()
        self.text_cnn    = text_cnn_encoder
        self.aggregation = aggregation_cell
        self.classifier  = nn.Linear(aggregation_cell.fc.out_features, 1)  # 最后一层全连接

    def forward(self, inputs, lengths, training=False):
        """
        inputs: shape = (总举报数, seq_length)
        lengths:每篇新闻包含的举报数
        返回:每篇新闻的概率
        """
        # 一次性通过 CNN 编码
        flat_feats = self.text_cnn(inputs)  # (总举报数, feat_dim)每条举报文本的特征向量

        # 聚合每篇新闻的举报特征
        agg_feats = self.aggregation(flat_feats, lengths, training=training)     

        # 最后一层 FC+Sigmoid 输出概率
        out = self.classifier(agg_feats)             # (batch, 1)
        out = torch.sigmoid(out)                     # 映射到 [0,1]
        return out
