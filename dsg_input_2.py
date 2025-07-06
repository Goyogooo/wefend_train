# Q
import json
import numpy as np
import jieba
import pandas as pd
from collections import defaultdict
import os


# [INFO] 基于Title确定max_seq_len: 23
# WEAK_TRAIN_PATH = "new-data/train_weak_data.json" 
LABELED_TRAIN_PATH = "new-data/train_split_80.csv" 

# WEAK_PROCESSED_PATH = "new-data/train_weak_processed.json"
LABELED_PROCESSED_PATH = "data2/labeled_train_processed.json"

MERGED_VOCAB_PATH = "data2/merged_title_vocab.json"  
VAL_PATH = "new-data/val_split_20.csv"
VAL_PROCESSED_PATH = "data2/val_labeled_processed.json"

##########
TEST_PATH = "data-all/data/test/news.csv"
TEST_PROCESSED_PATH = "data2/test_labeled_processed.json"

UNLABELED_PATH = "data-all/data/unlabeled data/news.csv"
UNLABELED_PROCESSED_PATH = "data2/unlabeled_processed.json"

VAL_WEAK_PATH="data2/new_val_set.json"
VAL_WEAK_PROCESSED_PATH="data2/val_weak_processed.json"

TRAIN_WEAK_DATA_PATH = "data2/train_weak_data.json"
TRAIN_WEAK_PROCESSED_PATH = "data2/train_weak_processed.json"

def load_weak_data(path):
    """加载弱标注数据（JSON格式），提取Title字段"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        if 'Title' not in item and 'title' in item:
            item['Title'] = item.pop('title')  
    return data

def load_labeled_data(path):
    """加载有标签数据（CSV格式），转换为字典列表"""
    df = pd.read_csv(path)
    if 'title' in df.columns:
        df = df.rename(columns={'title': 'Title'})
    data = df.to_dict('records')
    return data

def get_title_lengths(data):
    """仅统计Title的长度，用于确定max_seq_len"""
    lengths = []
    for item in data:
        title = str(item.get('Title', '')).strip()
        if title:
            tokens = jieba.lcut(title)
            lengths.append(len(tokens))
    return lengths

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
        if 'Report Content' in processed_item:
            processed_item['Report Content'] = str(processed_item['Report Content']).strip()
        elif 'reports' in processed_item:
            processed_item['reports'] = [str(r).strip() for r in processed_item['reports']]
        processed_data.append(processed_item)
    return processed_data

def collect_title_vocab(data):
    """仅从Title收集词汇，构建词表"""
    vocab = defaultdict(int)
    for item in data:
        title = str(item.get('Title', '')).strip()
        if title:
            tokens = jieba.lcut(title)
            for token in tokens:
                vocab[token] += 1
    return vocab


def write_title_dsg_corpus(processed_datasets, vocab, save_path):
    """生成Title的DSG输入文本（合并两类数据集的Title，用于嵌入初始化）"""
    inv_vocab = {v: k for k, v in vocab.items()}
    with open(save_path, "w", encoding="utf-8") as f:
        # 遍历弱标注和有标注数据集
        for dataset in processed_datasets:
            for item in dataset:
                # 提取处理后的Title ID序列（已padding）
                title_ids = item["Title"]
                # 转换ID为词汇（含<PAD>）
                title_tokens = [inv_vocab[id] for id in title_ids]
                # 写入一行（词汇用空格分隔）
                f.write(" ".join(title_tokens) + "\n")
    print(f"[保存] Title的DSG输入文件至 {save_path}，格式符合嵌入初始化要求")


if __name__ == "__main__":
    # 1. 加载两类原始数据
    weak_data = load_weak_data(TRAIN_WEAK_DATA_PATH)# 原始弱标注训练集
    labeled_data = load_labeled_data(LABELED_TRAIN_PATH)# 原始标注训练集
    val_data = load_labeled_data(VAL_PATH)# 原始标注验证集
    val_weak_data = load_weak_data(VAL_WEAK_PATH)# 原始弱标注验证集
    test_data = load_labeled_data(TEST_PATH)# 原始测试集

    # 2. 仅基于Title计算max_seq_len
    weak_title_lengths = get_title_lengths(weak_data)
    labeled_title_lengths = get_title_lengths(labeled_data)
    all_title_lengths = weak_title_lengths + labeled_title_lengths
    max_seq_len = int(np.percentile(all_title_lengths, 95)) if all_title_lengths else 10
    max_seq_len = max(max_seq_len, 10)  
    print(f"[INFO] 基于Title确定max_seq_len: {max_seq_len}")

    # 3. 合并两类数据的Title词汇，构建统一词表
    print("[INFO] 构建合并的Title词表...")
    weak_title_vocab = collect_title_vocab(weak_data)
    labeled_title_vocab = collect_title_vocab(labeled_data)
    all_title_vocab = {**weak_title_vocab,** labeled_title_vocab}
    # 添加特殊符号
    vocab_list = ["<PAD>", "<UNK>"] + [word for word in all_title_vocab.keys()]
    merged_vocab = {word: idx for idx, word in enumerate(vocab_list)}
    # 保存合并词表
    with open(MERGED_VOCAB_PATH, 'w', encoding='utf-8') as f:
        json.dump(merged_vocab, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 合并Title词表已保存至: {MERGED_VOCAB_PATH}，大小: {len(merged_vocab)}")

    # 4. 处理数据集
    print("[INFO] 处理数据集...")
    weak_processed = process_dataset(weak_data, max_seq_len, merged_vocab)
    labeled_processed = process_dataset(labeled_data, max_seq_len, merged_vocab)
    val_processed = process_dataset(val_data, max_seq_len, merged_vocab) 
    val_weak_processed = process_dataset(val_weak_data, max_seq_len, merged_vocab) 
    test_processed = process_dataset(test_data, max_seq_len, merged_vocab) 

    # 5. 分开保存处理后的两类数据集
    print("[INFO] 保存处理后的数据集...")
    # 保存弱标注训练集
    with open(TRAIN_WEAK_PROCESSED_PATH, 'w', encoding='utf-8') as f:
        json.dump(weak_processed, f, ensure_ascii=False, indent=2)
    # 保存有标注训练集
    with open(LABELED_PROCESSED_PATH, 'w', encoding='utf-8') as f:
        json.dump(labeled_processed, f, ensure_ascii=False, indent=2)
    # 保存验证集
    with open(VAL_PROCESSED_PATH, 'w', encoding='utf-8') as f:
        json.dump(val_processed, f, ensure_ascii=False, indent=2)
    with open(VAL_WEAK_PROCESSED_PATH, 'w', encoding='utf-8') as f:
        json.dump(val_weak_processed, f, ensure_ascii=False, indent=2)
    with open(TEST_PROCESSED_PATH, 'w', encoding='utf-8') as f:
        json.dump(test_processed, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] 弱标注训练集已保存至: {TRAIN_WEAK_PROCESSED_PATH}")
    print(f"[INFO] 有标注训练集已保存至: {LABELED_PROCESSED_PATH}")
    print(f"[INFO] 验证集已保存至: {VAL_PROCESSED_PATH}") 
    print(f"[INFO] 弱标注验证集已保存至: {VAL_WEAK_PROCESSED_PATH}") 
    print(f"[INFO] 测试集集已保存至: {TEST_PROCESSED_PATH}") 

    print("[INFO] 生成Title的DSG输入语料...")
    processed_datasets = [weak_processed, labeled_processed]
    write_title_dsg_corpus(processed_datasets, merged_vocab, "data2/title_dsg_corpus.txt")
    print("[INFO] 所有处理完成")