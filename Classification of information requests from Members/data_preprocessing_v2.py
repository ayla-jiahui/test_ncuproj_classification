import pandas as pd
import os
import json
from dotenv import load_dotenv
import ast
from glob import glob

load_dotenv()

cmd_path = r".\Classification of information requests from Members"
os.chdir(cmd_path)
print(os.getcwd())

# 讀取 raw_data 並清理空值
dir_list = os.listdir('./raw_data')
# print("找到的資料夾有：", dir_list)
folder_path = os.path.join('.', 'raw_data', dir_list[0])
# print("使用的資料夾路徑：", folder_path)
raw_data_path = os.getenv("RAW_DATA_PATH", folder_path)
# print("最終使用路徑：", raw_data_path)
raw_files = glob(os.path.join(raw_data_path, "*.xlsx"))
# print("找到的 Excel 檔案有：", raw_files)

for file in raw_files:
    # print(os.path.basename(file))  # 列出檔案名稱
    raw_file = os.path.basename(file)

    raw_data_file = os.path.join(raw_data_path, raw_file)
    print("\n使用的原始數據檔案：", raw_data_file)

    col_names =pd.read_excel(raw_data_file, nrows=0).columns.tolist()
    # print("欄位名稱：", col_names)

    raw_data = pd.read_excel(raw_data_file)
    print(f"原始數據有 {raw_data.shape[0]} 條。")
    raw_data_filter = raw_data[col_names[2:4] + [col_names[-1]]].dropna() # 只保留需要的欄位(['承辦機關', '質詢議題', '案件編號'])，並移除空值
    # print(raw_data_filter.head())
    print(f"移除空值後，剩 {raw_data_filter.shape[0]} 條原始數據。")

    # 合併前檢查：確保相同案件編號的「質詢議題」都一致
    conflict_check = raw_data_filter.groupby('案件編號')['質詢議題'].nunique() # 取得每個案件編號對應的「質詢議題」數量（nunique: 計算不同值的個數）
    conflicts = conflict_check[conflict_check > 1] # 表示同一案件有不同質詢議題
    if not conflicts.empty:
        print("以下案件編號的質詢議題不一致，請檢查：")
        print(conflicts)
        raise ValueError("有案件的質詢議題不一致，停止合併以避免錯誤。") # 為避免資料錯誤，直接中止程式。

    raw_data_filter['質詢議題'] = raw_data_filter['質詢議題'].str.replace('_x000D_', ' ', regex=False)\
                           .str.replace('\r', ' ', regex=False)\
                           .str.replace('\n', ' ', regex=False)\
                           .str.replace('\t', ' ', regex=False)\
                           .str.strip()

    # 合併相同的 編號，保留所有唯一的完整標籤
    # 對 raw_data 以 "編號" 欄位進行分組，as_index=False 表示分組後仍保留 "標號" 為欄位
    merged_data = raw_data_filter.groupby("案件編號", as_index=False).agg({
        "承辦機關": lambda x: list(x.unique()),
        '質詢議題': 'first', # 因已確認一致，取其中一個即可
    })
    print(merged_data.head(8))
    print(f"合併後剩 {merged_data.shape[0]} 條原始數據。") #Excel: SUMPRODUCT(1/COUNTIF(A2:A100, A2:A100))

    # 確保目錄存在
    os.makedirs("./merged_data", exist_ok=True)
    merged_name = f"{raw_file.split('.')[0]}_merged.csv"
    marged_file = os.path.join(".", "merged_data", merged_name)
    print(f"合併後的數據將儲存到：{marged_file}")

    try:
        # 儲存合併後的數據
        merged_data.to_csv(marged_file, index=False, encoding="utf-8-sig")
        print(f"合併後的數據已儲存!")
    except Exception as e:
        print(f"儲存合併後的數據時發生錯誤: {e}")