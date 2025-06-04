import pandas as pd
import os
import json
# from dotenv import load_dotenv
import ast
# from glob import glob
from collections import defaultdict
from tqdm.auto import tqdm
import random

# --- 配置 ---
cmd_path = r".\Classification of information requests from Members"
os.chdir(cmd_path)
print(f"當前工作目錄：{os.getcwd()}")

# 單標籤和多標籤數據的路徑
data_s = './merged_data/df_single.csv'
df_single_for_run = pd.read_csv(data_s)
data_m = './merged_data/df_multi.csv'
df_multi_for_run = pd.read_csv(data_m)

# 設定要處理的欄位名稱
TEXT_COLUMN_NAME = '質詢議題'
ID_COLUMN_NAME = '案件編號'
LABEL_SET_COLUMN_NAME = 'label_set'
NUM_SAMPLES_GOAL = 1 # 每個標籤目標樣本數

# === 解析標籤欄位，將其轉換為實際的 set 以便進行後續處理 ===
def parse_label_str(label_str):
    try:
        label_list = ast.literal_eval(label_str)
        return set(label.strip() for label in label_list if isinstance(label, str) and label.strip())
    except Exception:
        return set()
df_single_for_run['label_set'] = df_single_for_run['承辦機關'].fillna('').apply(parse_label_str)
df_multi_for_run['label_set'] = df_multi_for_run['承辦機關'].fillna('').apply(parse_label_str)

# 輸出路徑設定
output_base_path = "./sampled_data"
os.makedirs(output_base_path, exist_ok=True)

"""
    從給定的 DataFrame 中，為每個獨立出現的標籤抓取指定數量的樣本 (文本)，
    並確保每個被選用的文本ID在傳入的 `globally_used_ids` 集合中是唯一的。

    Args:
        dataframe_to_sample_from (pd.DataFrame): 從中抽樣的 DataFrame。
        num_samples_per_label_goal (int): 每個標籤希望抽取的樣本數量上限。
        text_column_name (str): 包含文本數據的列名。
        id_column_name (str): 包含唯一ID的列名。
        label_set_column_name (str): 包含標籤集合的列名。
        globally_used_ids (set):
            一個從外部傳入的集合，用於追蹤已經被選用過的文本ID。此函數會讀取並更新這個集合。
            這樣可以確保在多次調用此函數（即使針對不同的標籤或DataFrame子集，如果它們共享同一個 globally_used_ids 集合）時，同一個ID的文本不會被重複選取。
            在「單標籤」和「多標籤」獨立抽樣場景中，為它們分別傳入不同的集合。

    Returns:
        dict: {label_name (str): [sample_text_1 (str), sample_text_2 (str), ...]}
              只包含抽樣到的文本列表。
"""

# === 函數：為給定 DataFrame 中的每個獨立標籤抓取樣本和ID ===
def get_samples_with_ids_for_labels(
    dataframe_to_sample_from: pd.DataFrame,
    num_samples_per_label_goal: int,
    text_column_name: str,
    id_column_name: int,
    label_set_column_name: str,
    globally_used_ids: set
):
    # --- A. 初始檢查和準備 ---
    if dataframe_to_sample_from.empty:
        print("警告: 傳入的 DataFrame 為空，無法抽樣。")
        return {} # 直接返回空字典
    if not all(col in dataframe_to_sample_from.columns for col in [label_set_column_name, text_column_name, id_column_name]):
        print(f"錯誤: DataFrame 缺少必要的列: '{label_set_column_name}', '{text_column_name}', 或 '{id_column_name}'。")
        return {}

    # 獲取此 DataFrame 中所有獨立的標籤
    unique_labels_in_df = sorted(list(set(
        lbl for label_set in dataframe_to_sample_from[label_set_column_name] for lbl in label_set
    )))

    if not unique_labels_in_df:
        print("警告: 在 DataFrame 中沒有找到任何標籤進行抽樣。")
        return {}

    print(f"\n在 DataFrame (共 {len(dataframe_to_sample_from)} 行) 中找到 {len(unique_labels_in_df)} 個獨立標籤進行抽樣。")

    # 最終返回的，值是包含完整樣本信息的字典列表
    # {label_name: [{"id": ..., "text": ..., "original_label_set": ...}, ...]}
    label_to_selected_samples_info_dict = defaultdict(list)

    # 預先為每個標籤找到所有包含它的潛在樣本的 *DataFrame行索引*
    # 這樣我們可以稍後通過 .loc 獲取所有需要的列
    label_to_potential_df_indices = defaultdict(list)
    for df_idx, row_series in dataframe_to_sample_from.iterrows():
        for label_name_in_set in row_series[label_set_column_name]:
            label_to_potential_df_indices[label_name_in_set].append(df_idx) # 存儲行索引

    # 打亂標籤的處理順序，有助於在ID衝突時，不同標籤有較均等的機會獲得樣本
    random.shuffle(unique_labels_in_df)

    # --- B. 為每個標籤抽樣 ---
    for label_name_to_sample in tqdm(unique_labels_in_df, desc="為每個標籤抽樣 (ID去重)"):

        # 如果當前標籤的樣本已經抽夠了，則跳到下一個標籤
        if len(label_to_selected_samples_info_dict[label_name_to_sample]) >= num_samples_per_label_goal:
            continue

        # 獲取包含當前標籤的所有潛在樣本信息 (ID和文本)
        potential_df_indices_for_this_label = label_to_potential_df_indices[label_name_to_sample]

        # 打亂這些潛在樣本的順序，以增加抽樣的隨機性
        random.shuffle(potential_df_indices_for_this_label)

        for df_idx_to_use in potential_df_indices_for_this_label:
            if len(label_to_selected_samples_info_dict[label_name_to_sample]) < num_samples_per_label_goal:
                # 從 DataFrame 行中獲取 ID
                sample_id_to_check = dataframe_to_sample_from.loc[df_idx_to_use, id_column_name]

                if sample_id_to_check not in globally_used_ids:
                    # 如果 ID 未被使用過，則獲取所有需要的樣本信息
                    sample_text = dataframe_to_sample_from.loc[df_idx_to_use, text_column_name]
                    original_labels_for_this_sample = dataframe_to_sample_from.loc[df_idx_to_use, label_set_column_name] # 獲取原始標籤集

                    sample_info_to_add = {
                        "id": sample_id_to_check,
                        "text": sample_text,
                        "labels": list(original_labels_for_this_sample) # 轉換為列表以便JSON序列化
                    }

                    label_to_selected_samples_info_dict[label_name_to_sample].append(sample_info_to_add)
                    globally_used_ids.add(sample_id_to_check)
            else:
                break

    return dict(label_to_selected_samples_info_dict)

print("\n--- 從單一標籤數據 (df_single) 提取樣本 ---")
# 創建一個用於單標籤抽樣過程中的 ID 追蹤集合
ids_used_in_single_sampling = set()
if 'df_single_for_run' in locals() and not df_single_for_run.empty:
    single_label_samples = get_samples_with_ids_for_labels(
        dataframe_to_sample_from=df_single_for_run,
        num_samples_per_label_goal=NUM_SAMPLES_GOAL,
        text_column_name=TEXT_COLUMN_NAME,
        id_column_name=ID_COLUMN_NAME,
        label_set_column_name=LABEL_SET_COLUMN_NAME,
        globally_used_ids=ids_used_in_single_sampling # 傳入並允許被修改
    )
    output_single_path = os.path.join(output_base_path, "single_label_samples_with_global_id_dedup.json")
    with open(output_single_path, "w", encoding="utf-8") as f:
        json.dump(single_label_samples, f, ensure_ascii=False, indent=2)
    print(f"單一標籤抽樣結果已保存到: {output_single_path}")

else:
    print("df_single_for_run 未定義或為空，跳過單標籤抽樣。")
    single_label_samples = {}

print("\n--- 從多標籤數據 (df_multi) 提取樣本 ---")
ids_used_in_multi_sampling = set()
if 'df_multi_for_run' in locals() and not df_multi_for_run.empty:
    multi_label_samples = get_samples_with_ids_for_labels(
        dataframe_to_sample_from=df_multi_for_run,
        num_samples_per_label_goal=NUM_SAMPLES_GOAL,
        text_column_name=TEXT_COLUMN_NAME,
        id_column_name=ID_COLUMN_NAME,
        label_set_column_name=LABEL_SET_COLUMN_NAME,
        globally_used_ids=ids_used_in_multi_sampling # 傳入並允許被修改
    )
    output_multi_path = os.path.join(output_base_path, "multi_label_samples_with_global_id_dedup.json")
    with open(output_multi_path, "w", encoding="utf-8") as f:
        json.dump(multi_label_samples, f, ensure_ascii=False, indent=2)
    print(f"多標籤抽樣結果已保存到: {output_multi_path}")
else:
    print("df_multi_for_run 未定義或為空，跳過多標籤抽樣。")
    multi_label_samples = {}

# 如果你還需要一個最終合併所有標籤的字典，可以這樣做：
# final_all_samples = defaultdict(list)
# globally_used_ids_for_final_merge = set()

# # 優先級可以自己定，例如先放 multi_label_samples 的
# for label, samples_info_list in multi_label_samples_with_ids.items(): # 假設這個返回的是 {"id":..., "text":...}
#     for sample_info in samples_info_list:
#         if len(final_all_samples[label]) < NUM_SAMPLES_GOAL and sample_info['id'] not in globally_used_ids_for_final_merge:
#             final_all_samples[label].append(sample_info['text'])
#             globally_used_ids_for_final_merge.add(sample_info['id'])

# for label, samples_info_list in single_label_samples_with_ids.items():
#     for sample_info in samples_info_list:
#         if len(final_all_samples[label]) < NUM_SAMPLES_GOAL and sample_info['id'] not in globally_used_ids_for_final_merge:
#             final_all_samples[label].append(sample_info['text'])
#             globally_used_ids_for_final_merge.add(sample_info['id'])
# print("\n最終合併後的樣本（如果需要合併）：", dict(final_all_samples))