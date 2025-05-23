from collections import Counter
import os
import pandas as pd
from dotenv import load_dotenv
import ast

# 讀取 .env 環境變數（如果有設定的話）
load_dotenv()

path = r".\Classification of information requests from Members"
os.chdir(path)
print(os.getcwd())

#設定file from path

lable_file = "merged_final_11405.csv"
label_final_file = "label_counts_11405.csv"

# ==============================================
merged_data_path = os.getenv("LABEL_STRUCTURE_PATH", "./merged_data")
merged_data_file = os.path.join(merged_data_path, lable_file)
merged_data = pd.read_csv(merged_data_file, converters={"final_label": ast.literal_eval})
print(merged_data.head())

# 統計所有不重複的單一 label
all_labels = set(label for labels in merged_data["final_label"] for label in labels)
print(f"不重複的 label 數量：{len(all_labels)}")
print("所有 label：", all_labels)

# 將所有 final_label 展平成一個 list
all_labels = [label for labels in merged_data["final_label"] for label in labels]

# 統計每個標籤出現次數
label_counts = Counter(all_labels)

# 轉成 DataFrame 並排序
label_df = pd.DataFrame(label_counts.items(), columns=["label", "count"])
label_df = label_df.sort_values(by="count", ascending=False)

# 儲存成 CSV
output_path = os.path.join(merged_data_path, label_final_file)
label_df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"標籤統計已儲存到：{output_path}")
