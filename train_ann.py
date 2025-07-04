import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report
from aggregation_cell import AggregationCell
from annotator import Annotator
from text_cnn_encoder import TextCNNEncoder
import os

VOCAB_PATH        = "new-data/vocab.json"
EMBEDDING_PATH    = "new-data/dsg_embedding.npy"
TRAIN_JSON_PATH   = "new-data/train_grouped.json"
VAL_JSON_PATH     = "new-data/val_grouped.json"
SEQ_LENGTH        = 58
BATCH_SIZE        = 100
EPOCHS            = 100
LEARNING_RATE     = 1e-4
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 20
def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_grouped_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    inputs = [item["reports"] for item in data]
    labels = [item["label"] for item in data]
    return inputs, labels

# 处理原数据集单个新闻成为三元组，此新闻的所有举报，此新闻的标签，此新闻的举报数量
class NewsDataset(Dataset):
    def __init__(self, grouped_inputs, labels):
        self.flat_inputs = [report for reports in grouped_inputs for report in reports]
        self.lengths = [len(reports) for reports in grouped_inputs]
        self.labels = labels
        self.seq_length = SEQ_LENGTH

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        start = sum(self.lengths[:idx])# 定位当前新闻的举报范围
        end = start + self.lengths[idx]
        reports = self.flat_inputs[start:end]
        reports = [r + [0] * (self.seq_length - len(r)) if len(r) < self.seq_length else r[:self.seq_length] for r in reports]# 统一序列长度
        return torch.tensor(reports, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float), self.lengths[idx]

# 合并一个批次的新闻数据成为三元组，所有新闻的举报，所有新闻的标签，所有新闻的举报数
def collate_fn(batch):
    flat_reports = []
    labels = []
    lengths = []
    for reports, label, length in batch:
        flat_reports.extend(reports)
        labels.append(label)
        lengths.append(length)
    return torch.stack(flat_reports), torch.tensor(labels), torch.tensor(lengths)

def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for x, y, lens in dataloader:
            x, y, lens = x.to(DEVICE), y.to(DEVICE), lens.to(DEVICE)
            preds = model(x, lens).squeeze()
            all_probs.extend(preds.cpu().numpy())
            all_preds.extend((preds >= 0.55).int().cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)

def main():
    # 加载词典与嵌入
    vocab = load_vocab(VOCAB_PATH)
    embedding_matrix = np.load(EMBEDDING_PATH)
    vocab_size, embedding_dim = embedding_matrix.shape

    # 构建模型
    text_cnn = TextCNNEncoder(
        vocab_size, embedding_dim, embedding_matrix,
        filter_sizes=[1, 2, 3, 4, 5, 6], num_filters=40, feature_dim=40
    )
    agg_cell = AggregationCell(output_dim=20)
    model = Annotator(text_cnn, agg_cell).to(DEVICE)

    # 数据加载
    train_inputs, train_labels = load_grouped_json(TRAIN_JSON_PATH)
    val_inputs, val_labels = load_grouped_json(VAL_JSON_PATH)
    train_set = NewsDataset(train_inputs, train_labels)
    val_set = NewsDataset(val_inputs, val_labels)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 类别权重
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(train_labels),
                                                      y=np.array(train_labels))
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    criterion = nn.BCELoss(reduction='none')
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    best_f1 = 0
    count = 0
    # 训练模型
    for epoch in range(EPOCHS):
        model.train()
        loss = 0
        for x, y, lens in train_loader:
            x, y, lens = x.to(DEVICE), y.to(DEVICE), lens.to(DEVICE)
            optimizer.zero_grad()
            preds = model(x, lens).squeeze()
            weights = torch.where(y == 1, class_weights[1], class_weights[0])
            loss_tensor = criterion(preds, y) * weights
            batch_loss = loss_tensor.mean()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
        print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {loss/len(train_loader):.4f}")

        # 验证评估
        preds_bin, preds_prob, y_true = evaluate(model, val_loader)
        current_f1 = f1_score(y_true, preds_bin)
        print("Accuracy:", accuracy_score(y_true, preds_bin))
        print("AUC-ROC: ", roc_auc_score(y_true, preds_prob))
        print(classification_report(y_true, preds_bin, digits=4))
        print(f"Current F1 Score: {current_f1:.4f}")

        if current_f1 > best_f1:
            best_f1 = current_f1
            count = 0
            os.makedirs("saved_models", exist_ok=True)
            torch.save(model.state_dict(), "saved_models/best_annotator.pt")
        else:
            count += 1
            print(f"验证集 F1 未提升，等待轮数: {count}/{PATIENCE}")
            if count >= PATIENCE:
                print("[INFO] 早停触发，验证集 F1 不再提升")
                model.load_state_dict(torch.load("saved_models/best_annotator.pt", map_location=DEVICE))
                break
    
    model.load_state_dict(torch.load("saved_models/best_annotator.pt"))

if __name__ == "__main__":
    main()



# Accuracy: 0.8457169900615239
# AUC-ROC:  0.9190369860777501
#               precision    recall  f1-score   support

#          0.0     0.9329    0.8530    0.8912      1565
#          1.0     0.6628    0.8248    0.7350       548

