import pandas as pd
import numpy as np
import os

def process_all_tubes(
    source_dir=r"/home/tlmsq/Anomaly-Detection-Prediction/dataset",
    train_ratio=0.5
):
    """
    批量处理 tube1~tube5.csv 文件：
    1. 在 source_dir 下为每个 tubeX 创建同名子文件夹
    2. 重命名时间戳和特征列
    3. 按比例划分训练/测试集（默认50:50）
    4. 定义异常:
       a) 原始第17或第18列非零
    5. 保存 train.csv、test.csv、test_label.csv 至对应子文件夹
    6. 输出原始、训练和测试集中的异常数量
    """
    # 逐个处理 tube1~tube5
    for i in range(1, 6):
        base = f"tube{i}"
        source_path = os.path.join(source_dir, f"{base}.csv")
        if not os.path.isfile(source_path):
            print(f"⚠️ 跳过，未找到文件：{source_path}")
            continue

        # 创建 tubeX 子文件夹
        tube_dir = os.path.join(source_dir, base)
        os.makedirs(tube_dir, exist_ok=True)

        # 读取数据并重命名列
        df = pd.read_csv(source_path)
        df.rename(columns={df.columns[0]: 'timestamp_(min)'}, inplace=True)
        feature_cols = df.columns[1:16]
        feature_new_names = [f'feature_{j}' for j in range(len(feature_cols))]
        df.rename(columns=dict(zip(feature_cols, feature_new_names)), inplace=True)

        # 原始异常：第17或第18列非零
        orig_labels = ((df.iloc[:, 16] != 0) | (df.iloc[:, 17] != 0)).astype(int)

        # 提取最终特征数据（含 timestamp）
        export_cols = ['timestamp_(min)'] + feature_new_names
        df_features = df[export_cols]

        # 合并异常标签
        labels = orig_labels.reset_index(drop=True)
        original_anomalies = labels.sum()

        # 划分训练/测试集
        total_len = len(df_features)
        train_len = int(total_len * train_ratio)
        train_df = df_features.iloc[:train_len]
        test_df = df_features.iloc[train_len:]
        train_labels = labels.iloc[:train_len]
        test_labels = labels.iloc[train_len:]
        train_anomalies = train_labels.sum()
        test_anomalies = test_labels.sum()

        # 保存结果
        train_df.to_csv(os.path.join(tube_dir, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(tube_dir, 'test.csv'), index=False)
        test_label_df = pd.DataFrame({
            'timestamp_(min)': test_df['timestamp_(min)'].values,
            'label': test_labels
        })
        test_label_df.to_csv(os.path.join(tube_dir, 'test_label.csv'), index=False)

        # 输出统计信息
        print(f"✅ {base} 处理完成：")
        print(f"   - 原始样本数：{total_len} 行，异常数：{original_anomalies} 条")
        print(f"   - train.csv：{len(train_df)} 行，异常数：{train_anomalies} 条")
        print(f"   - test.csv：{len(test_df)} 行，异常数：{test_anomalies} 条")
        print(f"   - test_label.csv：{len(test_label_df)} 行")

# 脚本入口
if __name__ == "__main__":
    process_all_tubes()
