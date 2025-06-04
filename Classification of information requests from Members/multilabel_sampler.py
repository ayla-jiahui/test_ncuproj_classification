import pandas as pd
import os
import json
from dotenv import load_dotenv
import ast # 用來將字串安全轉換成 list
from glob import glob
from collections import defaultdict

load_dotenv()

cmd_path = r".\Classification of information requests from Members"
os.chdir(cmd_path)
print(f"當前工作目錄：{os.getcwd()}")
merged_data_path = os.getenv("MERGED_DATA_PATH", "./merged_data")
merged_files = glob(os.path.join(merged_data_path , "*.csv"))
normalized_paths = [p.replace("\\", "/") for p in merged_files]
print("找到的合併檔案有：", normalized_paths)

df_list = [pd.read_csv(file) for file in normalized_paths]
df = pd.concat(df_list, ignore_index=True)
print(f"合併檔案中共計 {df.shape[0]} 條。")

# Labels 欄位通常是字串格式的 list，例如 "['label1', 'label2']"，需要將其轉換為實際的 list 以便進行後續處理
def parse_label_str(label_str):
    try:
        label_list = ast.literal_eval(label_str)
        return set(label.strip() for label in label_list if isinstance(label, str) and label.strip())
    except Exception:
        return set()

# 統計所有獨立的 labels
df['label_set'] = df['承辦機關'].fillna('').apply(parse_label_str)
all_labels = set(label for labels in df['label_set'] for label in labels)
print(f"共出現 {len(all_labels)} 個獨立標籤")

df['label_count'] = df['label_set'].apply(len)
df_single = df[df['label_count'] == 1].copy()      # 只有一個 label
df_multi = df[(df['label_count'] > 1) & (df['label_count'] <= 9)].copy() # 多於一個 label，但少於 10 個
df_many  = df[df['label_count'] >= 10].copy()      # 有 10 個以上 label

print(f"單一標籤樣本數：{len(df_single)}")
print(f"多於一個標籤樣本數：{len(df_multi)}")
print(f"超過10個標籤樣本數：{len(df_many)}")

selected_columns = ["案件編號", "質詢議題", "承辦機關"]
df_single = df_single[selected_columns]
df_multi = df_multi[selected_columns]
df_many = df_many[selected_columns]

df_multi.to_csv("./merged_data/df_multi.csv", index=False, encoding='utf-8-sig')
df_single.to_csv("./merged_data/df_single.csv", index=False, encoding='utf-8-sig')
df_many.to_csv("./merged_data/df_many.csv", index=False, encoding='utf-8-sig')

# # 建立標籤對應的資料列 index 的字典
# def build_label_index_map(dataframe):
#     # 使用 defaultdict 來建立一個標籤對應的資料列 index 的字典
#     label_to_indices = defaultdict(list)
#     # 遍歷每一列，將標籤與其對應的 index 存入字典
#     for idx, row in dataframe.iterrows():
#         for label in row['label_set']:
#             label_to_indices[label].append(idx)
#     return label_to_indices

# import random

# def sample_examples(label_to_indices, df, n=3):
#     sampled = {}
#     for label, indices in label_to_indices.items():
#         if len(indices) >= n:
#             sampled[label] = df.loc[random.sample(indices, n)]
#         else:
#             sampled[label] = df.loc[indices]  # 少於3筆則全部保留
#     return sampled

# # 分別處理三類
# single_samples = sample_examples(build_label_index_map(df_single), df_single)