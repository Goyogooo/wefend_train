import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import os
import random
from selector import Selector, build_selector_state
from text_cnn_encoder import TextCNNEncoder
from itertools import cycle
import copy

VOCAB_PATH     = "data2/merged_title_vocab.json"
EMBEDDING_PATH = "data2/title_dsg.npy"
LABELED_PATH   = "data2/labeled_train_processed.json"
VAL_PATH       = "data2/val_labeled_processed.json"
WEAK_PATH      = "data2/train_weak_processed.json"
SEQ_LENGTH     = 23
BATCH_SIZE     = 100
EPOCHS         = 100
LEARNING_RATE  = 1e-4
BAG_SIZE       = 100
NUM_BAGS       = 200
BETA           = 1.0
PATIENCE       = 10
os.makedirs("saved_models", exist_ok=True)


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 加载词表
def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 加载有标签数据
def load_labeled_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    inputs = [item["Title"] for item in data]
    labels = [item["label"] for item in data]
    return inputs, labels

# 加载弱标签训练数据
def load_weak_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 创建PyTorch数据加载器
def make_dataloader(inputs, labels, batch_size, shuffle=True):
    inputs_tensor = torch.tensor(inputs, dtype=torch.long).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
    dataset = TensorDataset(inputs_tensor, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class FakeNewsDetector(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix):
        super().__init__()
        # 嵌入层
        self.encoder = TextCNNEncoder(
            vocab_size, embedding_dim, embedding_matrix,
            filter_sizes=[1,2,3,4,5,6], num_filters=40, feature_dim=40
        ).to(device)
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(40, 1),
            nn.Sigmoid()
        ).to(device)
    
    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x)


def train_detector_with_selector():
    vocab = load_vocab(VOCAB_PATH)
    embedding_matrix = np.load(EMBEDDING_PATH)
    vocab_size, embedding_dim = embedding_matrix.shape

    train_inputs, train_labels = load_labeled_data(LABELED_PATH)
    val_inputs, val_labels = load_labeled_data(VAL_PATH)
    weak_data = load_weak_data(WEAK_PATH)


    val_loader = make_dataloader(val_inputs, val_labels, batch_size=BATCH_SIZE, shuffle=False)

    # 1. 选择器初始化
    selector = Selector().to(device)
    selector_optimizer = optim.Adam(selector.parameters(), lr=LEARNING_RATE)

    # 2. 全局第0次训练检测器
    detector = FakeNewsDetector(vocab_size, embedding_dim, embedding_matrix).to(device)
    detector_optimizer = optim.Adam(detector.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    train_loader = make_dataloader(train_inputs, train_labels, BATCH_SIZE) 
    b_best_val_acc = 0
    b_patience_counter = 0
    for epoch in range(EPOCHS):
        detector.train()
        for inputs, labels in train_loader:
            labels = labels.view(-1, 1)
            detector_optimizer.zero_grad()
            outputs = detector(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            detector_optimizer.step()
    
        # 计算基础acc
        detector.eval() # 设置为评估模式
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = detector(inputs)
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy())
        
        baseline_preds = np.array(all_preds)
        b_baseline_acc = accuracy_score(np.array(all_labels), (baseline_preds >= 0.5).astype(int))

        if b_baseline_acc > b_best_val_acc:
            b_best_val_acc = b_baseline_acc
            b_patience_counter = 0
        else:
            b_patience_counter += 1
            if b_patience_counter >= PATIENCE:
                print(f"[Early Stop] baseline 第 {epoch+1} 轮停止，best_acc={b_best_val_acc:.4f}")
                break
    baseline_acc = b_best_val_acc
    print(f"[初始基线模型] 验证集 accuracy: {baseline_acc:.4f}")
    
    # 保存第0次训练检测器模型
    torch.save(detector.state_dict(), "saved_models/detector2_round_0.pt")

    all_best_f1 = 0
    all_patience_counter = 0
    # 主训练循环
    for round in range(10):
        print(f"\n[第 {round+1} 轮选择器+检测器训练]")

        # 5.1 选择器第1次采样用于训练网络
        total_sample_size = NUM_BAGS * BAG_SIZE  # 一次性采样 NUM_BAGS × BAG_SIZE 个不重复样本
        assert total_sample_size <= len(weak_data), "样本数量不足，无法不重复采样"
        all_indices = np.random.permutation(len(weak_data))[:total_sample_size]  # 随机选择索引并打乱
        bag_indices_list = np.split(all_indices, NUM_BAGS)  # 分成 K 个 bag，每个 bag 大小为 BAG_SIZE

        # 5.2 K个包处理
        for i, bag_indices in enumerate(bag_indices_list):
            if i % 2 == 1:
                continue

            bag = [weak_data[i] for i in bag_indices]
            bag_states, bag_actions = [], []
            bag_selected_inputs, bag_selected_labels = [], []  # 每个包最终被保留的结果

            detector.eval()  # 设置为评估模式
            selector.eval()  # 设置为评估模式
            with torch.no_grad():
                # 5.2.1 逐个新闻处理
                for item in bag:
                    title_ids = item['Title']
                    weak_label = item['weak_label']
                    p_annotator = item['pred_prob']  # 来自弱标注器的预测概率

                    xi = torch.tensor([title_ids], dtype=torch.long).to(device)
                    with torch.no_grad():
                        xi_embed = detector.encoder(xi)[0]  # CNN 提取特征表示
                        p_detector = detector(xi).item()  # 检测器当前预测值
                    
                    state = build_selector_state(xi_embed, bag_states, p_annotator, p_detector, weak_label)  # 构造状态向量用于策略网络输入
                    prob = selector(state.unsqueeze(0)).item()  # 强化选择器选择概率
                    action = np.random.rand() < prob  # 依据概率决定是否保留，使用伯努利分布采样动作

                    bag_states.append(state.detach().cpu().numpy())
                    bag_actions.append(float(action))
                    if action:
                        bag_selected_inputs.append(title_ids)
                        bag_selected_labels.append(weak_label)

            
            # 5.2.2 重新训练基础模型判断性能提升，（只用于计算新 reward）
            reward_model = copy.deepcopy(detector)
            reward_optimizer = optim.Adam(reward_model.parameters(), lr=LEARNING_RATE)
            # 创建数据加载器
            sup_loader = make_dataloader(train_inputs, train_labels, BATCH_SIZE)
            pseudo_loader = make_dataloader(bag_selected_inputs, bag_selected_labels, BATCH_SIZE)
            
            len_sup = len(sup_loader)
            len_pseudo = len(pseudo_loader)
            num_batches = max(len_sup, len_pseudo)
            # 训练奖励模型
            
            best_val_acc = 0
            patience_counter = 0
            
            for epoch in range(EPOCHS):
                reward_model.train()
                sup_iter = cycle(sup_loader) if len_sup < len_pseudo else iter(sup_loader)
                pseudo_iter = cycle(pseudo_loader) if len_pseudo < len_sup else iter(pseudo_loader)
                
                for _ in range(num_batches):
                    x_sup, y_sup = next(sup_iter)
                    x_pseudo, y_pseudo = next(pseudo_iter)
                    
                    y_sup = y_sup.view(-1, 1)
                    y_pseudo = y_pseudo.view(-1, 1)
                    
                    reward_optimizer.zero_grad()
                    
                    # 有标签损失
                    y_pred_sup = reward_model(x_sup)
                    loss_sup = criterion(y_pred_sup, y_sup)
                    
                    # 伪标签损失
                    y_pred_pseudo = reward_model(x_pseudo)
                    loss_pseudo = criterion(y_pred_pseudo, y_pseudo)
                    
                    # 总损失
                    total_loss = loss_sup + BETA * loss_pseudo
                    total_loss.backward()
                    reward_optimizer.step()

                # 5.2.3 验证集计算新acc
                reward_model.eval()
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        outputs = reward_model(inputs)
                        all_preds.extend(outputs.cpu().numpy().flatten())
                        all_labels.extend(labels.cpu().numpy())
                
                val_acc_k = accuracy_score(np.array(all_labels), (np.array(all_preds) >= 0.5).astype(int))

                if val_acc_k > best_val_acc:
                    best_val_acc = val_acc_k
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        # print(f"[Early Stop] reward_model 第 {epoch+1} 轮停止，best_acc={best_val_acc:.4f}")
                        break

            reward_k = best_val_acc - baseline_acc  # 当前轮性能提升作为奖励
            # print(f"[选择后训练] acc_k={best_val_acc:.4f}, baseline_acc={baseline_acc:.4f}, reward={reward_k:.4f}")

            # 5.2.4 更新选择器策略网络
            selector.train()
            selector_optimizer.zero_grad()
            
            # 准备数据
            states_tensor = torch.stack([torch.tensor(s, dtype=torch.float32).to(device) for s in bag_states])
            actions_tensor = torch.tensor(bag_actions, dtype=torch.float32).to(device)
            
            # 计算损失
            probs = selector(states_tensor)
            probs = torch.clamp(probs, 1e-6, 1.0)
            log_probs = torch.log(probs)
            reinforce_loss = -torch.mean(log_probs.squeeze() * actions_tensor * reward_k)
            
            reinforce_loss.backward()
            selector_optimizer.step()
            print(f"[Selector] loss={reinforce_loss.item():.6f} ")
        
        # 5.3 保存选择器结果
        torch.save(selector.state_dict(), f"saved_models/selector2_round_{round+1}.pt")

        # 5.4 选择器第2次采样，选择检测器输入样本
        re_total_sample_size = NUM_BAGS * BAG_SIZE  # 一次性采样 NUM_BAGS × BAG_SIZE 个不重复样本
        assert re_total_sample_size <= len(weak_data), "样本数量不足，无法不重复采样"
        re_all_indices = np.random.permutation(len(weak_data))[:re_total_sample_size]  # 随机选择索引并打乱
        re_bag_indices_list = np.split(re_all_indices, NUM_BAGS)  # 分成 K 个 bag，每个 bag 大小为 BAG_SIZE

        selected_inputs, selected_labels = [], []  # 保存K个包的高质量筛选结果

        for re_bag_indices in re_bag_indices_list:
            bag = [weak_data[i] for i in re_bag_indices]
            bag_states, bag_actions = [], []
            
            # 5.4.1 逐个新闻处理
            for item in bag:
                title_ids = item['Title']
                weak_label = item['weak_label']
                p_annotator = item['pred_prob']  # 来自弱标注器的预测概率

                xi = torch.tensor([title_ids], dtype=torch.long).to(device)
                with torch.no_grad():
                    xi_embed = detector.encoder(xi)[0]  # CNN 提取特征表示
                    p_detector = detector(xi).item()  # 检测器当前预测值
                
                state = build_selector_state(xi_embed, bag_states, p_annotator, p_detector, weak_label)  # 构造状态向量用于策略网络输入
                prob = selector(state.unsqueeze(0)).item()  # 强化选择器选择概率
                action = np.random.rand() < prob  # 依据概率决定是否保留，使用伯努利分布采样动作

                bag_states.append(state.detach().cpu().numpy())
                bag_actions.append(float(action))
                if action:
                    selected_inputs.append(title_ids)
                    selected_labels.append(weak_label)

        # 5.5 检测器训练
        sup_loader = make_dataloader(train_inputs, train_labels, BATCH_SIZE)
        pseudo_loader = make_dataloader(selected_inputs, selected_labels, BATCH_SIZE) 
        detector_optimizer = optim.Adam(detector.parameters(), lr=LEARNING_RATE)
        
        len_sup = len(sup_loader)
        len_pseudo = len(pseudo_loader)
        num_batches = max(len_sup, len_pseudo)

        d_best_f1 = 0
        d_patience_counter = 0
        tmp_acc = 0
        for epoch in range(EPOCHS):
            detector.train()
            sup_iter = cycle(sup_loader) if len_sup < len_pseudo else iter(sup_loader)
            pseudo_iter = cycle(pseudo_loader) if len_pseudo < len_sup else iter(pseudo_loader)

            for _ in range(num_batches):
                x_sup, y_sup = next(sup_iter)
                x_pseudo, y_pseudo = next(pseudo_iter)
                
                y_sup = y_sup.view(-1, 1)
                y_pseudo = y_pseudo.view(-1, 1)
                
                detector_optimizer.zero_grad()
                
                # 有标签损失
                y_pred_sup = detector(x_sup)
                loss_sup = criterion(y_pred_sup, y_sup)
                
                # 伪标签损失
                y_pred_pseudo = detector(x_pseudo)
                loss_pseudo = criterion(y_pred_pseudo, y_pseudo)
                
                # 总损失
                total_loss = loss_sup + BETA * loss_pseudo
                total_loss.backward()
                detector_optimizer.step()
            
            detector.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = detector(inputs)
                    all_preds.extend(outputs.cpu().numpy().flatten())
                    all_labels.extend(labels.cpu().numpy())
            
            d_preds_prob = np.array(all_preds)
            d_preds_bin = (d_preds_prob >= 0.5).astype(int)
            d_y_true = np.array(all_labels)
            val_f1=f1_score(d_y_true, d_preds_bin, pos_label=1)
            val_acc = accuracy_score(d_y_true, d_preds_bin)
            if val_f1 > d_best_f1:
                d_best_f1 = val_f1
                d_patience_counter = 0
                tmp_acc = val_acc
            else:
                d_patience_counter += 1
                if d_patience_counter >= PATIENCE:
                    print(f"[Early Stop] detector 第 {epoch+1} 轮停止，best_f1={d_best_f1:.4f}")
                    break


        # 大循环判断
        if d_best_f1 > all_best_f1:
            all_best_f1 = d_best_f1
            all_patience_counter = 0
            baseline_acc = tmp_acc
            torch.save(detector.state_dict(), f"saved_models/wefend_detector2.pt")
        else:
            all_patience_counter += 1
            if all_patience_counter >= 3:
                print(f"[Early Stop] detector 第 {epoch+1} 轮停止，best_f1={d_best_f1:.4f}")
                detector.load_state_dict(torch.load("saved_models/wefend_detector2.pt"))
                break
 
    # 使用验证集评估
    detector.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = detector(inputs)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy())
    
    d_preds_prob = np.array(all_preds)
    d_preds_bin = (d_preds_prob >= 0.5).astype(int)
    d_y_true = np.array(all_labels)

    # 输出评估指标
    print("Accuracy:", accuracy_score(d_y_true, d_preds_bin))
    print("AUC-ROC: ", roc_auc_score(d_y_true, d_preds_prob))
    print(classification_report(d_y_true, d_preds_bin, digits=4))

if __name__ == "__main__":
    train_detector_with_selector()

# Accuracy: 0.9636449480642115
# AUC-ROC:  0.9868501859223735
#               precision    recall  f1-score   support

#          0.0     0.9746    0.9764    0.9755      1569
#          1.0     0.9322    0.9271    0.9297       549
