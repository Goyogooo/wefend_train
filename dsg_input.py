import pandas as pd
import jieba
import json
import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split


ORIGINAL_TRAIN_CSV = "data-all/data/train/news.csv"  # 原始训练集
TEST_CSV = "data-all/data/test/news.csv"              # 原始测试集

SAVED_TRAIN_CSV = "train_split_80.csv"  # 80%新训练集
SAVED_VAL_CSV = "val_split_20.csv"      # 20%验证集

VOCAB_OUT = "vocab.json"
TRAIN_DSG_TXT = "train_dsg_corpus.txt"  # 仅训练集需要DSG语料
TRAIN_GROUPED_JSON = "train_grouped.json"
VAL_GROUPED_JSON = "val_grouped.json"
TEST_GROUPED_JSON = "test_grouped.json"

PERCENTILE = 95 
RANDOM_STATE = 42  



def tokenize(text):
    """中文分词（复用jieba）"""
    return list(jieba.cut(str(text).strip()))

def analyze_lengths(token_lists, percentile=95):
    """统计token序列长度，返回指定分位数的长度作为max_seq_len"""
    lengths = [len(tokens) for tokens in token_lists if tokens]
    if not lengths:
        return 0
    max_len = int(np.percentile(lengths, percentile))
    print(f"[长度分析] 样本数: {len(lengths)}, 最大: {np.max(lengths)}, 平均: {np.mean(lengths):.2f}, {percentile}%分位数: {max_len}")
    return max_len

def build_vocab(all_tokens):
    """基于训练集tokens构建词表（含<PAD>和<UNK>）"""
    counter = Counter()
    for tokens in all_tokens:
        counter.update(tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, _ in counter.most_common(): 
        vocab[word] = idx
        idx += 1
    print(f"[词表] 大小（含PAD/UNK）: {len(vocab)}")
    return vocab

def tokens_to_ids(tokens, vocab, max_seq_len):
    """将tokens转换为ID，截断/补全至max_seq_len"""
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    if len(ids) > max_seq_len:
        return ids[:max_seq_len]  # 截断
    else:
        return ids + [vocab["<PAD>"]] * (max_seq_len - len(ids))  # 补全

def read_tokens_from_df(df):
    """从DataFrame中读取report并分词"""
    all_tokens = []
    for _, row in df.iterrows():
        reports = str(row.get("Report Content", "")).split("##")
        for report in reports:
            report = report.strip()
            if report:
                all_tokens.append(tokenize(report))
    return all_tokens

def save_grouped_json(df, vocab, max_seq_len, save_path):
    """保存分组后的JSON（每条新闻含多个report的ID序列）"""
    grouped_data = []
    for idx, row in df.iterrows():
        reports = str(row.get("Report Content", "")).split("##")
        encoded_reports = []
        for report in reports:
            report = report.strip()
            if report:
                tokens = tokenize(report)
                encoded = tokens_to_ids(tokens, vocab, max_seq_len)
                encoded_reports.append(encoded)
        if encoded_reports:  # 仅保留有有效report的样本
            grouped_data.append({
                "news_id": f"news_{idx}",
                "reports": encoded_reports,
                "label": int(row.get("label", 0))
            })
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(grouped_data, f, ensure_ascii=False, indent=2)
    print(f"[保存] 分组JSON至 {save_path}，样本数: {len(grouped_data)}")

def write_dsg_corpus(tokens_list, vocab, max_seq_len, save_path):
    """生成训练集的DSG输入文本"""
    inv_vocab = {v: k for k, v in vocab.items()}
    with open(save_path, "w", encoding="utf-8") as f:
        for tokens in tokens_list:
            ids = tokens_to_ids(tokens, vocab, max_seq_len)
            padded_tokens = [inv_vocab[id] for id in ids]
            f.write(" ".join(padded_tokens) + "\n")
    print(f"[保存] 训练集DSG语料至 {save_path}")


def main():
    # 1. 读取原始训练集并划分8:2，保存原始格式
    print("\n===== 步骤1: 划分并保存训练集与验证集（原始格式） =====")
    df_original = pd.read_csv(ORIGINAL_TRAIN_CSV)
    print(f"原始训练集总样本数: {len(df_original)}")
    # 分层抽样划分，保持label分布一致
    df_train, df_val = train_test_split(
        df_original, test_size=0.2, random_state=RANDOM_STATE, stratify=df_original["label"]
    )
    # 保存为原始CSV格式
    df_train.to_csv(SAVED_TRAIN_CSV, index=False, encoding="utf-8")
    df_val.to_csv(SAVED_VAL_CSV, index=False, encoding="utf-8")
    print(f"[保存] 新训练集（80%）至 {SAVED_TRAIN_CSV}，样本数: {len(df_train)}")
    print(f"[保存] 验证集（20%）至 {SAVED_VAL_CSV}，样本数: {len(df_val)}")

    # 2. 基于新训练集（80%）进行预处理
    print("\n===== 步骤2: 处理新训练集（80%） =====")
    # 读取划分后的训练集
    df_train_processed = pd.read_csv(SAVED_TRAIN_CSV)
    train_tokens = read_tokens_from_df(df_train_processed)
    # 确定max_seq_len
    max_seq_len = analyze_lengths(train_tokens, percentile=PERCENTILE)
    # 构建并保存词表
    vocab = build_vocab(train_tokens)
    with open(VOCAB_OUT, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"[保存] 词表至 {VOCAB_OUT}")
    # 生成训练集DSG语料和分组JSON
    write_dsg_corpus(train_tokens, vocab, max_seq_len, TRAIN_DSG_TXT)
    save_grouped_json(df_train_processed, vocab, max_seq_len, TRAIN_GROUPED_JSON)

    # 3. 处理验证集（复用训练集参数）
    print("\n===== 步骤3: 处理验证集 =====")
    df_val_processed = pd.read_csv(SAVED_VAL_CSV)
    save_grouped_json(df_val_processed, vocab, max_seq_len, VAL_GROUPED_JSON)

    # 4. 处理测试集（复用训练集参数）
    print("\n===== 步骤4: 处理测试集 =====")
    df_test = pd.read_csv(TEST_CSV)
    save_grouped_json(df_test, vocab, max_seq_len, TEST_GROUPED_JSON)

    print("\n===== 所有处理完成 =====")

if __name__ == "__main__":
    main()