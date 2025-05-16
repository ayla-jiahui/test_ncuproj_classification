from collections import Counter
import os
import pandas as pd
from dotenv import load_dotenv
import ast

# 讀取 .env 環境變數（如果有設定的話）
load_dotenv()

merged_data_path = os.getenv("LABEL_STRUCTURE_PATH", "./merged_data")
merged_data_file = f"{merged_data_path}/merged_data_0517.csv"
merged_data = pd.read_csv(merged_data_file, converters={"new_label": ast.literal_eval})

# 統計所有不重複的單一 label
all_labels = set(label for labels in merged_data["new_label"] for label in labels)
print(f"不重複的 label 數量：{len(all_labels)}")
print("所有 label：", all_labels)

# 將所有 new_label 展平成一個 list
all_labels = [label for labels in merged_data["new_label"] for label in labels]

# 統計每個標籤出現次數
label_counts = Counter(all_labels)

# 轉成 DataFrame 並排序
label_df = pd.DataFrame(label_counts.items(), columns=["label", "count"])
label_df = label_df.sort_values(by="count", ascending=False)

# 儲存成 CSV
output_path = f"{merged_data_path}/label_counts_0517.csv"
label_df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"標籤統計已儲存到：{output_path}")
