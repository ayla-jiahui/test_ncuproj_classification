import pandas as pd
import os
import json
from dotenv import load_dotenv
import ast

load_dotenv()

# 讀取 label_structure
label_structure_path = os.getenv("LABEL_STRUCTURE_PATH", "./label_structure")
label_structure_file = f"{label_structure_path}/label_structure.xlsx"
label_structure = pd.read_excel(label_structure_file)

# === 從 JSON 文件載入別名/縮寫對照表 ===
with open("alias_mapping.json", "r", encoding="utf-8") as f:
    alias_to_full_name = json.load(f)
    # print(type(alias_to_full_name))
print(f"成功載入別名對照表。")

# ===========================================

# 讀取 raw_data 並清理空值
raw_data_path = os.getenv("RAW_DATA_PATH", "./raw_data")
raw_data_file = f"{raw_data_path}/113索資_latest.xlsx"
raw_data = pd.read_excel(raw_data_file)[['索取資料題目', '承辦機關']]
raw_data.columns = ['text', 'original_label']
raw_data = raw_data.dropna()

# 合併相同的 text，保留所有唯一的完整標籤
merged_data = raw_data.groupby("text", as_index=False).agg({
    "original_label": lambda x: list(x.unique())
}) # 14782
print(merged_data.shape)

# 確保目錄存在
os.makedirs("./merged_data", exist_ok=True)
# merged_data.to_csv("./merged_data/merged_data_latest.csv", index=False, encoding="utf-8-sig")

# =======================================================

def preprocess_label_with_aliases(label, alias_map):
    #將輸入標籤通過別名表轉換為標準名稱 (如果存在於別名表中)
    return alias_map.get(label, label)

# === 新增：定義需要特殊保留二級的「特定一級機關」列表 ===
specific_primary_to_keep_secondary_set = set([
    "工務局",
    "都市發展局",
    "產業發展局"
])

# ========================================================

print("資料分類中......")

def _get_single_label_classification(ori_lbl_to_classify, label_structure_df, specific_primary_to_keep_secondary_set, alias_map):
    #輔助函數：為單個標籤找到分類結果

    lbl_to_classify = preprocess_label_with_aliases(ori_lbl_to_classify, alias_map)

    # 首先進行二級標籤比對
    for _, row in label_structure_df.iterrows():
        secondary = str(row["secondary"]) if not pd.isna(row["secondary"]) else ""
        primary = str(row["primary"]) if not pd.isna(row["primary"]) else ""

        if secondary and secondary in lbl_to_classify: # 確保 secondary 不是 None 也不是空字串
            # print(f"[二級比對] 原始 label: '{lbl_to_classify}' ➜ 比對到二級: '{secondary}' (其一級為: '{primary}')", end=' ')
            if primary in specific_primary_to_keep_secondary_set:
                # print("→ 保留")
                # print(primary+secondary)
                return primary+secondary
            else:
                classified_as = primary if primary else "其他"
                # print(f"→ 其一級不在特定列表，歸類為一級: '{primary if primary else '其他'}'")
                return classified_as

    # 如果二級沒有比對到，再進行一級標籤比對
    for _, row in label_structure_df.iterrows():
        primary = str(row["primary"]) if not pd.isna(row["primary"]) else ""
        if primary and primary in lbl_to_classify:
            #print(f"[一級比對] 原始 label: '{lbl_to_classify}' ➜ 比對到一級: '{primary}'")
            return primary

    # 如果一級和二級都沒有比對到，歸類為 "其他"
    print(f"[未比對] 原始 label: '{ori_lbl_to_classify}' (處理後: '{lbl_to_classify}') ➜ 歸類為 '其他'")
    return "其他"

def reclassify_labels_for_row(list_of_original_labels_for_a_text, label_structure_data, specific_primary_to_keep_secondary_set, alias_map_param):
    reclassified_for_this_text = []
    # 遍歷從 merged_data['original_label'] 傳來的原始標籤列表
    # 例如 list_of_original_labels_for_a_text 可能會是 ['文化局', '消防局']

    for single_original_lbl in list_of_original_labels_for_a_text:
        # 確保當前處理的是字串
        lbl_str = str(single_original_lbl)

        actual_labels_to_process = []
        # 檢查當前標籤字串是否是 "列表的字串表示" (例如, 字串 "['內政部', '外交部']")

        if lbl_str.startswith('[') and lbl_str.endswith(']'): # 這是為了處理原始數據中可能存在的髒數據或特殊格式
            # print(f"警告: 原始標籤 '{lbl_str}' 看起來像列表字串，嘗試解析。")
            try:
                parsed_list_inner = ast.literal_eval(lbl_str) # 嘗試將 "列表的字串表示" 安全地轉換為實際的 Python 列表
                if isinstance(parsed_list_inner, list): # 如果成功解析為列表，則將列表中的每個元素加入待分類列表
                    actual_labels_to_process.extend(map(str, parsed_list_inner))
                else: # 如果解析結果不是列表 (例如字串 "[abc]" 解析為字串 "[abc]")，則將其作為單個標籤
                    actual_labels_to_process.append(str(parsed_list_inner))
            except (ValueError, SyntaxError):  # 解析失敗，則將原始字串視為一個整體標籤
                actual_labels_to_process.append(lbl_str)
        else: # 如果不是 "列表的字串表示"，說明它本身就是一個單獨的標籤 (例如 '文化局')
            actual_labels_to_process.append(lbl_str)
        # 對於從上述步驟中提取出的每一個待分類標籤，進行實際的分類操作
        for lbl_item_to_classify in actual_labels_to_process:
            classification = _get_single_label_classification(lbl_item_to_classify, label_structure_data, specific_primary_to_keep_secondary_set, alias_map_param)
            reclassified_for_this_text.append(classification)

    # 對於當前行文本的所有分類結果，去重並保持順序
    return list(dict.fromkeys(reclassified_for_this_text))

# def reclassify_labels(label_list):
#     reclassified_labels = []
#     for lbl in label_list:
#         matched = False
#         for _, row in label_structure.iterrows():
#             secondary = str(row["secondary"]) if not pd.isna(row["secondary"]) else ""
#             primary = str(row["primary"]) if not pd.isna(row["primary"]) else ""
#             if secondary and secondary in lbl:
#                 reclassified_labels.append(secondary)
#                 matched = True
#                 break
#             if primary and primary in lbl:
#                 reclassified_labels.append(primary)
#                 matched = True
#                 break
#         if not matched:
#             reclassified_labels.append("其他")
#     # 直接回傳 list
#     return list(dict.fromkeys(reclassified_labels))


print("開始對合併後的數據進行標籤重分類...")
merged_data['reclassified_label'] = merged_data['original_label'].apply(
    lambda list_of_labels: reclassify_labels_for_row(list_of_labels, label_structure, specific_primary_to_keep_secondary_set, alias_to_full_name) # 使用別名對照表進行標籤重分類
)
print("標籤重分類完成。")

# 查看結果
print(merged_data.head())

# 儲存結果
# merged_data.to_csv("./merged_data/merged_data_0522.csv", index=False, encoding="utf-8-sig")

# ==========================================================
# 輸入為json檔案

# 讀取並預處理 outside JSON 數據
with open("1140428_民政部門質詢D1_簡報表.json", "r", encoding="utf-8") as f:
    outside_data = json.load(f)
print(f"\n成功讀取 {len(outside_data)} 條外部數據。")

for item in outside_data:
    original_labels = item.get("labels", [])
    print(f"原始標籤: {original_labels}")
    reclassified = reclassify_labels_for_row(
        original_labels,
        label_structure,
        specific_primary_to_keep_secondary_set,
        alias_to_full_name
    )
    item["reclassified_labels"] = reclassified
    print(f"重分類後的標籤: {reclassified}")

# 儲存結果
with open("140428_民政部門質詢D1_簡報表_WithReclassified.json", "w", encoding="utf-8") as f:
    json.dump(outside_data, f, ensure_ascii=False, indent=2)