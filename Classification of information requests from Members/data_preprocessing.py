import pandas as pd
import os

# 讀取 raw data 和 label_structure
raw_data_path = os.getenv("RAW_DATA_PATH", "./raw_data")
raw_data_file = f"{raw_data_path}/113索資.xlsx"

label_structure_path = os.getenv("LABEL_STRUCTURE_PATH", "./label_structure")
label_structure_file = f"{label_structure_path}/label_structure.xlsx"
label_structure = pd.read_excel(label_structure_file)

# 讀取 raw_data 並清理空值
raw_data = pd.read_excel(raw_data_file)[['索取資料題目', '承辦機關']]
raw_data.columns = ['text', 'label']
raw_data = raw_data.dropna()

# 合併相同的 text，保留所有唯一的完整標籤
merged_data = raw_data.groupby("text", as_index=False).agg({
    "label": lambda x: list(x.unique())
}) # 14783
# merged_data.to_csv("./data/merged_data.csv", index=False, encoding="utf-8-sig")

# 提取固定架構表中的一級和二級機關
valid_labels = set(label_structure["primary"]).union(set(label_structure["secondary"]))

print("資料整合中......")

def reclassify_labels(label_list):
    reclassified_labels = []
    for lbl in label_list:
        matched = False
        for _, row in label_structure.iterrows():
            secondary = str(row["secondary"]) if not pd.isna(row["secondary"]) else ""
            primary = str(row["primary"]) if not pd.isna(row["primary"]) else ""
            if secondary and secondary in lbl:
                reclassified_labels.append(secondary)
                matched = True
                break
            if primary and primary in lbl:
                reclassified_labels.append(primary)
                matched = True
                break
        if not matched:
            reclassified_labels.append("其他")
    # 直接回傳 list
    return list(dict.fromkeys(reclassified_labels))

# 新增一個欄位存放重新分類後的標籤
merged_data["new_label"] = merged_data["label"].apply(reclassify_labels)

# 確保目錄存在
os.makedirs("./merged_data", exist_ok=True)

# 儲存結果
merged_data.to_csv("./merged_data/merged_data_0516.csv", index=False, encoding="utf-8-sig")