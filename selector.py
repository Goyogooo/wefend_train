#Q
import torch
import torch.nn as nn
import torch.nn.functional as F

class Selector(nn.Module):
    """
    Reinforced Selector 网络
    用于从弱标注样本中选择高质量数据。

    输入：
        state (batch_size, 88)，由以下部分拼接而成：
            - 当前样本的 state 向量（TextCNN 表示 + 状态特征 = 44）
            - 平均已选样本的 state 向量（TextCNN 平均表示 + 平均状态特征 = 44）
    输出：
        selection_prob (batch_size, 1)：样本被选择的概率（0~1）
    """
    def __init__(self):
        super(Selector, self).__init__()
        # 线性层 + ReLU，输入是 88 维，输出是 8 维，提取非线性特征
        self.dense1 = nn.Linear(88, 8)  
        # 输出层，使用 sigmoid 激活将结果映射到 [0,1] 区间，表示该样本被选择的概率
        self.dense2 = nn.Linear(8, 1)   

    def forward(self, state):
        x = F.relu(self.dense1(state))     # (batch_size, 8)
        prob = torch.sigmoid(self.dense2(x))  # (batch_size, 1)
        return prob


# —— 状态构建函数 ——
def build_selector_state(xi_embed, selected_states, p_annotator, p_detector, weak_label):
    """
    构建 selector 的 88维 state 向量
    输入：
        xi_embed: 当前样本的 TextCNN 表示，形状 (40,)
        selected_states: 已选样本的完整 state 向量集合 [ (44,), (44,), ... ]
        p_annotator: float，注释器预测为假新闻的概率
        p_detector: float，检测器预测为假新闻的概率
        weak_label: int (0 或 1)
    输出：
        state 向量，形状为 (88,)
    """
    device = xi_embed.device

    # 当前样本的状态特征（4维）
    if selected_states:
        # 提取已选样本中的 TextCNN 表示部分（前 40）
        selected_embeds = [s[:40] for s in selected_states]
        # 确保在相同设备上计算余弦相似度
        sims = []
        for s in selected_embeds:
            # 将历史状态转换为与当前嵌入相同的设备和数据类型
            s_tensor = torch.tensor(s, dtype=torch.float32, device=device)
            sim = F.cosine_similarity(xi_embed, s_tensor, dim=0).item()
            sims.append(sim)
        max_sim = -min(sims)  # 越相似越小，取负最大
    else:
        max_sim = 0.0

    current_state_feat = torch.tensor([
        float(p_annotator),
        float(p_detector),
        float(max_sim),
        float(weak_label)
    ], dtype=torch.float32, device=device)  # (4,)

    current_state = torch.cat([xi_embed, current_state_feat], dim=0)  # (44,)

    if selected_states:
        selected_tensor = torch.stack(
            [torch.tensor(s[:44], dtype=torch.float32, device=device) 
             for s in selected_states], dim=0
        )
        avg_state = torch.mean(selected_tensor, dim=0)  # (44,)
    else:
        avg_state = torch.zeros(44, dtype=torch.float32, device=device)

    final_state = torch.cat([current_state, avg_state], dim=0)  # (88,)
    return final_state
