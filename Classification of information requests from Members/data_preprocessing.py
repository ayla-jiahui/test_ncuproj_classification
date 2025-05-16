import pandas as pd
import os
import glob

# 讀取 raw data 和 label_structure
raw_data_path = os.getenv("RAW_DATA_PATH", "./raw_data")
raw_data_file = f"{raw_data_path}/113索資.xlsx"

label_structure_path = os.getenv("LABEL_STRUCTURE_PATH", "./label_structure")
label_structure_file = f"{label_structure_path}/label_structure.xlsx"
label_structure = pd.read_excel(label_structure_file)

# 讀取 raw_data 並清理空值
raw_data = pd.read_excel(raw_data_file)[['索取資料題目', '承辦機關']] # 23222
raw_data.columns = ['text', 'label']
# removed_data = raw_data[raw_data["label"].isna()] # 25
raw_data = raw_data.dropna() # 23197

# 合併相同的 text
merged_data = raw_data.groupby("text", as_index=False).agg({
    "label": lambda x: ",".join(x.unique())  # 合併相同 text 的 label，去重後用逗號分隔
}) # 14783
# merged_data.to_csv("./data/merged_data.csv", index=False, encoding="utf-8-sig")

# 提取固定架構表中的一級和二級機關
valid_labels = set(label_structure["primary"]).union(set(label_structure["secondary"]))

print("資料整合中......")
# 定義一個函式，用於處理每筆資料的標籤
def reclassify_labels(label):
    # 拆分標籤
    labels = [lbl.strip() for lbl in label.split(",") if lbl.strip()]
    # print(f"Classifying label: {label}")  # 調試輸出

    reclassified_labels = []

    # 優先匹配二級機關
    for lbl in labels:
        matched = False
        for _, row in label_structure.iterrows():
            secondary = str(row["secondary"]) if not pd.isna(row["secondary"]) else ""
            primary = str(row["primary"]) if not pd.isna(row["primary"]) else ""

            # 優先匹配二級機關
            if secondary and secondary in lbl:
                reclassified_labels.append(secondary)
                # print(f"Matched secondary: {secondary}")
                matched = True
                break

            # 匹配一級機關
            if primary and primary in lbl:
                reclassified_labels.append(primary)
                # print(f"Matched primary: {primary}")
                matched = True
                break

        # 如果無法匹配，標記為 "其他"
        if not matched:
            # print("No match found, returning '其他'")
            reclassified_labels.append("其他")

    # 去重並重新組合
    return ", ".join(dict.fromkeys(reclassified_labels))  # 使用 dict.fromkeys 保留順序

# 新增一個欄位存放重新分類後的標籤
merged_data["new_label"] = merged_data["label"].apply(reclassify_labels)

# 保留原始標籤和新標籤對比
# print(merged_data.head())
merged_data.to_csv("./merged_data/merged_data_with_labels.csv", index=False, encoding="utf-8-sig")