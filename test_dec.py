import json
import numpy as np
from detector import FakeNewsDetector
import jieba
import pandas as pd
from collections import defaultdict
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from text_cnn_encoder import TextCNNEncoder


# [INFO] 基于Title确定max_seq_len: 23
MERGED_VOCAB_PATH = "new-data/merged_title_vocab.json"  
EMBEDDING_PATH = "new-data/title_dsg.npy"

TEST_PATH = "data-all/data/test/news.csv"
TEST_PROCESSED_PATH = "new-data/test_labeled_processed.json"

# MODEL_PATH = "checkpoints/semi_cnn_detector.pt"
MODEL_PATH = "checkpoints/weak_detector.pt"
BASE_MODEL_PATH = "checkpoints/detector_round_0.pt"

UNLABELED_PATH = "data-all/data/unlabeled data/news.csv"
UNLABELED_PROCESSED_PATH = "new-data/unlabeled_processed.json"

VAL_WEAK_PATH="data2/new_val_set.json"
VAL_WEAK_PROCESSED_PATH="data2/val_weak_processed.json"

TRAIN_WEAK_DATA_PATH = "data2/train_weak_data.json"
TRAIN_WEAK_PROCESSED_PATH = "data2/train_weak_processed.json"

SEQ_LENGTH = 23
BATCH_SIZE = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载词表
def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 原始csv格式
def load_labeled_data(path):
    """加载有标签数据（CSV格式），转换为字典列表"""
    df = pd.read_csv(path)
    # 处理字段名大小写，统一为'Title'
    if 'title' in df.columns:
        df = df.rename(columns={'title': 'Title'})
    # 转换为字典列表，保留所有原始字段
    data = df.to_dict('records')
    return data

# json 包含weak_label or label
def load_val_weak_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data  # 保留所有字段，包括 weak_label 和 pred_prob

# 加载有标签数据
def load_labeled_title_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    inputs = [item["Title"] for item in data]
    labels = [item["label"] for item in data]
    return inputs, labels

def process_title(title, max_seq_len, vocab=None):
    """仅对Title进行分词、截断/补全（padding），可选转为ID"""
    title_str = str(title).strip()
    tokens = jieba.lcut(title_str) if title_str else []
    # 截断过长
    if len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]
    # 补全过短（仅对Title用<PAD>）
    pad_len = max_seq_len - len(tokens)
    tokens += ["<PAD>"] * pad_len
    # 若提供vocab则转为ID
    if vocab is not None:
        tokens = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    return tokens

def process_dataset(data, max_seq_len, vocab=None):
    """处理数据集：仅对Title进行padding，保留report原始值"""
    processed_data = []
    for item in data:
        processed_item = item.copy()  # 保留所有原始字段
        # 处理Title（仅对Title进行padding）
        title = item.get('Title', '')
        processed_item['Title'] = process_title(title, max_seq_len, vocab)
        # 不处理report，保留原始值（若存在）
        if 'Report Content' in processed_item:
            processed_item['Report Content'] = str(processed_item['Report Content']).strip()
        elif 'reports' in processed_item:
            processed_item['reports'] = [str(r).strip() for r in processed_item['reports']]
        processed_data.append(processed_item)
    return processed_data


def make_dataloader(inputs, labels, batch_size, shuffle=False):
    """创建PyTorch数据加载器"""
    inputs_tensor = torch.tensor(inputs, dtype=torch.long).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
    dataset = TensorDataset(inputs_tensor, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



def save_results_to_file(results, model_path, test_size, seq_length, save_path="test_results.txt"):
    """将评估指标保存到文件"""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"模型性能评估: {model_path}\n")
        f.write("=" * 60 + "\n")
        f.write(f"测试集大小: {test_size}\n")
        f.write(f"序列长度: {seq_length}\n")
        f.write(f"准确率 (Accuracy): {results['accuracy']:.4f}\n")
        f.write(f"AUC-ROC: {results['auc_roc']:.4f}\n")
        f.write(f"F1分数: {results['f1_score']:.4f}\n")
        f.write("\n分类报告:\n")
        f.write(results['classification_report'])
    print(f"\n评估结果已保存到 {save_path}")

def process(path,processed_path):
    print("[INFO] 加载数据...")
    
    path_data = load_labeled_data(path)
    max_seq_len = SEQ_LENGTH
    vocab = load_vocab(MERGED_VOCAB_PATH)
    path_processed = process_dataset(path_data, max_seq_len, vocab)
    print("[INFO] 保存处理后的数据集...")
    with open(processed_path, 'w', encoding='utf-8') as f:
        json.dump(path_processed, f, ensure_ascii=False, indent=2)

def process_val_weak(path,processed_path):
    print("[INFO] 加载数据...")
    
    path_data = load_val_weak_data(path)
    max_seq_len = SEQ_LENGTH
    vocab = load_vocab(MERGED_VOCAB_PATH)
    path_processed = process_dataset(path_data, max_seq_len, vocab)
    print("[INFO] 保存处理后的数据集...")
    with open(processed_path, 'w', encoding='utf-8') as f:
        json.dump(path_processed, f, ensure_ascii=False, indent=2)
    
# ===================== 测试评估函数 =====================
def evaluate_model(model, test_loader):
    """在测试集上评估模型性能"""
    model.eval()  # 设置为评估模式
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            
            # 收集预测结果
            all_probs.extend(outputs.cpu().numpy().flatten())
            all_preds.extend((outputs >= 0.55).int().cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy())
    
    # 转换为numpy数组
    probs = np.array(all_probs)
    preds = np.array(all_preds)
    true_labels = np.array(all_labels)
    
    # 计算评估指标
    accuracy = accuracy_score(true_labels, preds)
    auc_roc = roc_auc_score(true_labels, probs)
    f1 = f1_score(true_labels, preds)
    report = classification_report(true_labels, preds, digits=4)
    
    # 打印结果
    print("=" * 60)
    print(f"模型性能评估: {MODEL_PATH}")
    print("=" * 60)
    print(f"测试集大小: {len(true_labels)}")
    print(f"序列长度: {SEQ_LENGTH}")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"F1分数: {f1:.4f}")
    print("\n分类报告:")
    print(report)
    
    # 返回结果
    return {
        "accuracy": accuracy,
        "auc_roc": auc_roc,
        "f1_score": f1,
        "classification_report": report
    }

# ===================== 主函数 =====================
def test_wrapper(model_path):
    
    embedding_matrix = np.load(EMBEDDING_PATH)
    vocab_size, embedding_dim = embedding_matrix.shape

    model = FakeNewsDetector(vocab_size, embedding_dim, embedding_matrix).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型: {model_path}")
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 {model_path}")
        return
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return
    
    test_inputs, test_labels = load_labeled_title_data(TEST_PROCESSED_PATH)
    
    test_loader = make_dataloader(test_inputs, test_labels, BATCH_SIZE)
    print(f"测试集大小: {len(test_inputs)}")

    # 5. 评估模型
    results = evaluate_model(model, test_loader)
    
    save_results_to_file(
        results,
        model_path=model_path,
        test_size=len(test_inputs),
        seq_length=SEQ_LENGTH,
        save_path=(f"{model_path}_results2.txt")
    )


# -------------------------- 主流程 --------------------------
if __name__ == "__main__":
    # path = UNLABELED_PATH
    # processed_path = UNLABELED_PROCESSED_PATH
    # process(path,processed_path)
    # model_path = MODEL_PATH
    # test_wrapper(model_path)
    process_val_weak(TRAIN_WEAK_DATA_PATH,TRAIN_WEAK_PROCESSED_PATH)
    process_val_weak(VAL_WEAK_PATH,VAL_WEAK_PROCESSED_PATH)
