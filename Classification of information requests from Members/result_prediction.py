import json
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    BertForSequenceClassification,TrainingArguments,
    Trainer, TrainerCallback, EvalPrediction, AutoTokenizer)
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, hamming_loss, classification_report
import jieba
import re

from google.colab import drive
drive.mount("/content/drive")

''' 步驟 1: 資料路徑 '''
# 設定 Google Drive 路徑
google_drive_path = "/content/drive/MyDrive/Colab Notebooks/test"

# 設定模型、MultiLabelBinarizer 的路徑
model_path = os.path.join(google_drive_path, f"RoBERTa/0602/model_regularization_jieba/checkpoint-69608")
mlb_classes_json_path = os.path.join(google_drive_path, f"RoBERTa/mlb_classes.json")

# 輸出結果的路徑
output_result_path = os.path.join(google_drive_path, f"RoBERTa/predict")
output_base_path = os.path.join(output_result_path,"test")
os.makedirs(output_base_path, exist_ok=True)

# 最終儲存所有評估指標的 JSON 路徑
metrics_path = os.path.join(output_base_path, "all_experiment_metrics.json")

# ===== 步驟2: 讀取外部 JSON 數據_v1 =====
# outside_data_path = os.path.join(google_drive_path,"140428_民政部門質詢D1_簡報表_WithReclassified.json")

all_input_items = [] # 儲存原始 JSON 對象
texts_for_prediction = [] # 儲存text
true_labels_original_format = [] # 儲存label

# 讀取並預處理 outside JSON 數據
with open(outside_data_path, 'r', encoding='utf-8') as f:
  outside_data_json = json.load(f)
  for item in tqdm(outside_data_json, desc="Processing outside data"):
    if "text" in item and "labels" in item:  # 確保兩個關鍵欄位都存在
    # 如果有 "不分辦"，直接跳過這筆資料（下面的 append 都不執行）
      # if any("不分辦" in lbl for lbl in item.get("reclassified_labels", [])):
      #   continue
      # print(f"ID: {item['id']}, Text: {item['text']}")
      all_input_items.append(item)
      texts_for_prediction.append(item["text"])
      true_labels_original_format.append(item["reclassified_labels"])

print(f"\n成功讀取 {len(texts_for_prediction)} 條外部數據。")
print(all_input_items[11])

# ===== 步驟2: 讀取外部 JSON 數據_v2 =====
# outside_data_path = os.path.join(google_drive_path, f"RoBERTa/single_label_samples_with_reclassified.json")
# outside_data_path = os.path.join(google_drive_path, f"RoBERTa/multi_label_samples_with_reclassified.json")

all_input_items = [] # 儲存原始 JSON 對象
texts_for_prediction = [] # 儲存text
true_labels_original_format = [] # 儲存label
id_list = [] # 儲存id

# 讀取並預處理 outside JSON 數據
with open(outside_data_path, 'r', encoding='utf-8') as f:
  outside_data_json = json.load(f)
  for label_name, samples_list in outside_data_json.items():
    # print(f"\n標籤: {label_name}")
    if samples_list: # 確保樣本列表不為空
        for i, sample_info in enumerate(samples_list):
            all_input_items.append(sample_info)
            id_list.append(sample_info["id"])
            texts_for_prediction.append(sample_info["text"])
            true_labels_original_format.append(sample_info["reclassified_labels"])
    else:
        print(f"標籤-{label_name}: 無抽樣樣本")

print(f"成功讀取 {len(texts_for_prediction)} 條外部數據。")
print(all_input_items[0])

''' ==== 步驟2: 讀取外部 JSON 數據_v3 ===== 這是從原本的訓練模型資料裡抽取的測試集，不用經過以下斷詞、和移除贅字 '''

#outside_data_path = os.path.join(google_drive_path, f"RoBERTa/test_texts_Re_jieba.json")
all_input_items = [] # 儲存原始 JSON 對象
texts_for_prediction = [] # 儲存text
true_labels_original_format = [] # 儲存label

# 讀取並預處理 outside JSON 數據
with open(outside_data_path, 'r', encoding='utf-8') as f:
  outside_data_json = json.load(f)
  for item in tqdm(outside_data_json, desc="Processing outside data"):
    if "text" in item and "labels" in item:  # 確保兩個關鍵欄位都存在
      all_input_items.append(item)
      texts_for_prediction.append(item["text"])
      true_labels_original_format.append(item["labels"].split(','))

print(f"\n成功讀取 {len(texts_for_prediction)} 條外部數據。")
print(true_labels_original_format[10])

# ===== 前處理-移除贅詞 =====
redundant_words = [
    "有", "是", "與", "的", "在", "從", "到", "自", "由", "往", "朝", "於", "對", "跟", "與", "給", "為",
    "因", "因為", "由於", "由", "因應", "靠", "以", "用", "替", "為了", "關於", "對於", "就", "有關",
    "自從", "直到", "至", "依", "據", "憑", "循", "藉由", "藉", "趁", "比", "如", "按照", "根據", "依照",
    "照", "沿著", "沿", "順著", "隨", "隨著", "除了", "除了以外", "不如", "］", "?", "？", "［", "。",
    "提供", "制表", "彙整後", "貴單位", "主旨", "：", ":", "懇請", "（", "）", "一、", "二、", "三、", "四、",
    "五、", "六、", "1.", "2.", "3.", "4.", "5.", "1、", "2、", "3、", "4、", "5、", "承上", "敬請", "茲問政需要，",
    "惠予", "_x000D_", "惠請", "【", "】"
]

redundant_words = sorted(redundant_words, key=lambda x: -len(x))
# print(redundant_words)

def remove_redundant_words(text):
    for word in redundant_words:
        text = text.replace(word, "")
    return text.strip()
for i in range(len(texts_for_prediction)):
    texts_for_prediction[i] = remove_redundant_words(texts_for_prediction[i])

print(f"移除後: {texts_for_prediction[0]}")

# 當 'text' 欄位有編號
pattern = r'\b[A-Za-z]\d{5}-\d{8}\b' # 英文字母 + 5個數字 + "-" + 8個數字
for i in range(len(texts_for_prediction)):
    if re.search(pattern, texts_for_prediction[i]):
      print(texts_for_prediction[i])
      texts_for_prediction[i] = re.sub(pattern, '', texts_for_prediction[i])
      texts_for_prediction[i] = texts_for_prediction[i].strip()
print(f"移除編號後: {texts_for_prediction[0]}")

# ===== 前處理-jieba斷詞 =====
texts_for_prediction = [jieba.lcut(text) for text in texts_for_prediction]
print(f"斷詞: {texts_for_prediction[0]}")
texts_for_prediction = [" ".join(text) for text in texts_for_prediction]
print(f"斷詞後整合: {texts_for_prediction[0]}")

# ============== 加載 MultiLabelBinarizer 和 Tokenizer ==============
with open(mlb_classes_json_path, "r", encoding="utf-8") as f:
      classes_list = json.load(f)
mlb = MultiLabelBinarizer()
mlb.fit([classes_list])
classes_list = mlb.classes_
print(f"MLB fit 完成。\nmlb.classes_: {mlb.classes_}")
print(f"\nMLB fitted classes count: {len(mlb.classes_)}")

true_labels_binarized_np = mlb.transform(true_labels_original_format)
print(f"\n二值化後的真實標籤形狀: {true_labels_binarized_np.shape}")

# 加載 Tokenizer
try:
    # tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(google_drive_path, f"RoBERTa/0602/model_regularization_jieba"))
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("\nTokenizer 加載成功。")
except Exception as e:
    print(f"錯誤: 無法加載 Tokenizer: {e}")
    exit()

# ================= 檢查 CUDA 是否可用並加載模型 ================
# 檢查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# 加載模型
try:
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval() # 設置為評估模式
except Exception as e:
    print(f"\n錯誤: {e}");exit()

# ================= 檢查模型和 MultiLabelBinarizer 的類別數量 ================
# 驗證 MLB 的類別數量是否與模型的輸出層匹配
num_model_labels = model.config.num_labels
print(f"\n模型訓練時有 {num_model_labels} 個標籤")
num_mlb_classes = len(mlb.classes_)
print(f"用於預測的 MLB 有 {num_mlb_classes} 個類別")

if num_model_labels != num_mlb_classes:
    print(f"\n警告: 模型訓練時有 {num_model_labels} 個標籤，但 MLB fit時有 {num_mlb_classes} 個類別。")
    print("這可能表示模型與用於預測的 MLB 不匹配。")
    # 如果錯誤仍然存在，或者評估指標看起來不對勁，你可能需要調查原因。
else:
    print(f"\n模型標籤數量 ({num_model_labels}) 與 MLB 類別數量 ({num_mlb_classes}) 匹配。")

# ================= 定義 MultiLabelDataset 類別 =================
class MultiLabelDataset(Dataset):
    def __init__(self, texts_list, labels_binarized_np, tokenizer_instance, max_seq_length):
        self.texts = texts_list
        print(f"Tokenizing {len(texts_list)} texts for inference dataset...")

        self.encodings = tokenizer_instance(
            list(tqdm(texts_list, desc="Tokenizing")),
            truncation=True, #截斷
            padding=True, #填白
            max_length=max_seq_length,
            return_tensors='pt'   # 返回 PyTorch Tensors
        )

        if not isinstance(labels_binarized_np, torch.Tensor):
            self.true_labels_tensor = torch.tensor(labels_binarized_np, dtype=torch.float)
        else:
            self.true_labels_tensor = labels_binarized_np.float() # 確保是 float

        print("\nInference dataset initialization complete.")

    def __getitem__(self, idx): #可以用索引方式取得某一筆資料
        # 從已經是 Tensor 的 self.encodings 中獲取對應索引的數據
        # .clone().detach() 確保返回的是一個新的 tensor 副本
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}

        # 添加對應索引的真實標籤
        item['labels'] = self.true_labels_tensor[idx].clone().detach()
        return item

    def __len__(self): #回傳資料集的長度
        return len(self.encodings['input_ids'])

# --- 步驟 3: 準備測試數據集 ---
test_dataset = MultiLabelDataset(texts_for_prediction, true_labels_binarized_np, tokenizer, max_seq_length=512)
print(f"\ntest_dataset: {test_dataset}")

# ================= 定義預測模型的函數 =================
def run_inference(dataset_instance, model_instance, batch_size_val: int = 16, device_str: str = None):

  if device_str:
    current_device = torch.device(device_str)
  else:
    current_device = next(model_instance.parameters()).device

  print(f"正在使用設備: {current_device}")

  model_instance.eval() # 確保模型在評估模式
  model_instance.to(current_device) # 確保模型在指定設備

  # 使用 DataLoader 批次處理
  dataloader = DataLoader(dataset_instance, batch_size=batch_size_val, shuffle=False)

  all_logits_list, all_true_labels_list = [], []

  print(f"開始對數據進行推理 (共 {len(dataset_instance)} 條)...")

  with torch.no_grad():
    for batch in tqdm(dataloader, desc="Running inference batches"):
      input_ids = batch['input_ids'].to(current_device)
      attention_mask = batch['attention_mask'].to(current_device)

      labels_from_batch = batch['labels']
      all_true_labels_list.append(labels_from_batch.cpu())

      model_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
      outputs = model_instance(**model_inputs)
      all_logits_list.append(outputs.logits.cpu())

  if not all_logits_list:
    print("警告: 推理過程中沒有產生 logits。")
    return None, None

  logits_all_np = torch.cat(all_logits_list, dim=0).numpy()
  true_labels_all_binarized_np = torch.cat(all_true_labels_list, dim=0).numpy().astype(int)
  probs_all_np = torch.sigmoid(torch.from_numpy(logits_all_np).float()).numpy()

  return probs_all_np, true_labels_all_binarized_np

# --- 步驟 4: 獲取預測概率和真實標籤 ---
all_test_probs_np, all_test_true_labels_np = run_inference(
    dataset_instance=test_dataset,
    model_instance=model,
    batch_size_val=16,
    device_str=None
)

def get_set_from_label_names_list(label_names_list):
  if not label_names_list or not isinstance(label_names_list, (list, tuple)): # 是 None、空、不是 list/tuple，就直接回傳空集合 set()
    return set()
  return set(s.strip() for s in label_names_list if isinstance(s, str) and s.strip())

def evaluate_and_save_with_threshold(
    id_list,
    probs_all_np,
    true_labels_all_binarized_np,
    original_texts_list,
    mlb_instance,
    prediction_threshold: float,
    actual_output_csv_filename: str,
    top_k_count: int = 3):

  if probs_all_np is None or true_labels_all_binarized_np is None:
    print("錯誤: 概率或真實標籤為 None，無法進行評估。")
    return {}

  num_samples = len(probs_all_np)
  if len(original_texts_list) != num_samples or len(true_labels_all_binarized_np) != num_samples:
    print("警告: 輸入數據的樣本數量不一致！CSV 可能不完整或指標基於不匹配數據。")

  print(f"\n--- 使用閾值 {prediction_threshold} 進行評估和保存 ---")

  preds_binary_threshold = (probs_all_np >= prediction_threshold).astype(int)

  print(f"評估指標 (閾值 {prediction_threshold}):")
  metrics_report_dict = classification_report(
      true_labels_all_binarized_np,
      preds_binary_threshold,
      target_names=mlb_instance.classes_,
      zero_division=0,
      output_dict=True
  )
  micro_f1 = f1_score(true_labels_all_binarized_np, preds_binary_threshold, average='micro', zero_division=0)
  macro_f1 = f1_score(true_labels_all_binarized_np, preds_binary_threshold, average='macro', zero_division=0)
  subset_accuracy = (preds_binary_threshold == true_labels_all_binarized_np).all(axis=1).mean()

  intersection = (preds_binary_threshold & true_labels_all_binarized_np).sum(axis=1)
  union = ((preds_binary_threshold | true_labels_all_binarized_np).sum(axis=1))
  sample_jaccard_scores = intersection / (union + 1e-8) # 避免除以零
  mean_sample_jaccard = np.mean(sample_jaccard_scores) if len(sample_jaccard_scores) > 0 else 0.0
  micro_precision = precision_score(true_labels_all_binarized_np, preds_binary_threshold, average="micro", zero_division=0)
  micro_recall = recall_score(true_labels_all_binarized_np, preds_binary_threshold, average="micro", zero_division=0)

  print(f"Micro F1 (Thresh): {micro_f1:.6f}")
  print(f"Macro F1 (Thresh): {macro_f1:.6f}")
  print(f"Subset Accuracy (Thresh): {subset_accuracy:.6f}")
  print(f"Sample Accuracy (Thresh): {mean_sample_jaccard:.6f}")
  print(f"Micro_precison: {micro_precision:.6f}")
  print(f"Micro_recall: {micro_recall:.6f}")

  # 準備要返回的聚合指標字典
  current_metrics = {
      f"micro_f1_thresh_{prediction_threshold}": round(micro_f1, 6),
      f"macro_f1_thresh_{prediction_threshold}": round(macro_f1, 6),
      f"subset_accuracy_thresh_{prediction_threshold}": round(subset_accuracy, 6),
      f"sample_accuracy_thresh_{prediction_threshold}": round(mean_sample_jaccard, 6),
      f"micro_precision_thresh_{prediction_threshold}": round(micro_precision, 6),
      f"micro_recall_thresh_{prediction_threshold}": round(micro_recall, 6)
  }

  csv_output_data = []
  print(f"\n準備 CSV 數據並進行逐樣本詳細分析 (共 {num_samples} 條)...")

  # 先將所有二值化標籤轉換回標籤名稱列表
  all_true_labels_names = mlb_instance.inverse_transform(true_labels_all_binarized_np)
  all_predicted_labels_thresh_names = mlb_instance.inverse_transform(preds_binary_threshold)

  for i in range(len(original_texts_list)):
    text = original_texts_list[i]
    prob_row = probs_all_np[i]

    true_names_this_sample_tuple = all_true_labels_names[i]
    pred_names_thresh_this_sample_tuple = all_predicted_labels_thresh_names[i]

    true_set = get_set_from_label_names_list(list(true_names_this_sample_tuple))
    pred_set_thresh = get_set_from_label_names_list(list(pred_names_thresh_this_sample_tuple))
    # print(pred_set_thresh)

    # 計算當前樣本基於閾值的 TP, FP, FN
    tp_thresh_set = pred_set_thresh.intersection(true_set)
    fp_thresh_set = pred_set_thresh.difference(true_set)
    fn_thresh_this_sample_set = true_set.difference(pred_set_thresh)

    # Top-k 預測
    top_k_indices_csv = np.argsort(prob_row)[::-1][:top_k_count]
    top_k_names_csv = [mlb_instance.classes_[idx] for idx in top_k_indices_csv]
    top_k_probs_csv = [prob_row[idx] for idx in top_k_indices_csv]
    pred_names_top_k_set = get_set_from_label_names_list(top_k_names_csv)

    # 分析被閾值忽略但在 Top-K 中的「潛在正確」預測
    missed_by_thresh_but_in_top_k_set = fn_thresh_this_sample_set.intersection(pred_names_top_k_set)

    # 將閾值中的小數點 "." 轉成底線 "_"
    threshold_str = str(prediction_threshold).replace('.', '_')

    # 確保 top_k_names_csv 的長度 >= top_k_count，否則中止並提示錯誤
    assert len(top_k_names_csv) >= top_k_count, "top_k_names_csv 長度不足"
    # 確保 top_k_probs_csv 的長度 >= top_k_count，否則中止並提示錯誤
    assert len(top_k_probs_csv) >= top_k_count, "top_k_probs_csv 長度不足"

    # 是否完全匹配
    is_exact_match = (true_set == pred_set_thresh)

    # Top-K 是否命中至少一個正確答案
    topk_hit = bool(true_set.intersection(top_k_names_csv))

    # 預測狀態分類
    if is_exact_match:
        status = "完全匹配" if true_set else "空集合匹配"
    elif not pred_set_thresh and true_set:
        status = "預測為空，實際有標籤"
    elif pred_set_thresh and not true_set:
        status = "預測有標籤，實際為空"
    else:
        status = "部分正確或錯誤"

    status_parts = []
    if len(tp_thresh_set) > 0:
        status_parts.append(f"對 {len(tp_thresh_set)} 個")
    if len(fp_thresh_set) > 0:
        status_parts.append(f"錯 {len(fp_thresh_set)} 個")
    if len(fn_thresh_this_sample_set) > 0:
        status_parts.append(f"漏 {len(fn_thresh_this_sample_set)} 個")

    if not status_parts: # 理論上不應該到這裡，因為如果 TP, FP, FN 都為0，則應該是完全匹配
        return "部分正確/錯誤 (詳細情況未知)"
    detailed_status = "，".join(status_parts)

    record = {
        "案件編號": id_list[i],
        "質詢議題": text[:200] + "..." if len(text) > 200 else text,
        "實際承辦機關": ";".join(sorted(list(true_set))) if true_set else "",
        "預測承辦機關": ";".join(sorted(list(pred_set_thresh))) if pred_set_thresh else "",
        "預測狀態": status,
        "預測詳細狀態": detailed_status,
        # "精確度(Precision)": len(tp_thresh_set) / (len(tp_thresh_set) + len(fp_thresh_set)) if (tp_thresh_set or fp_thresh_set) else None,
        # "召回率(Recall)": len(tp_thresh_set) / (len(tp_thresh_set) + len(fn_thresh_this_sample_set)) if (tp_thresh_set or fn_thresh_this_sample_set) else None,
        # "完全匹配": is_exact_match,
        # "TopK命中": topk_hit,
        # "TP(答對)": len(tp_thresh_set),
        # "FP(誤判)": len(fp_thresh_set),
        # "FN(漏掉)": len(fn_thresh_this_sample_set),
        "漏掉的標籤": ";".join(sorted(list(fn_thresh_this_sample_set))) if fn_thresh_this_sample_set else "",
        "閾值漏掉，Top-K 撈回": ";".join(sorted(list(missed_by_thresh_but_in_top_k_set))) if missed_by_thresh_but_in_top_k_set else ""
        # "TopK彌補數": len(missed_by_thresh_but_in_top_k_set)
      }
    # print(record)
    for k_idx in range(top_k_count): # 添加 Top-K 結果到每一行
      record[f"Top{k_idx+1}_預測機關"] = top_k_names_csv[k_idx]
      record[f"Top{k_idx+1}_機率"] = f"{top_k_probs_csv[k_idx]:.4f}"

      # (可選) 添加所有類別的概率
      # for cls_idx, cls_name in enumerate(mlb_instance.classes_):
      #     record[f"prob_{cls_name}"] = f"{prob_row[cls_idx]:.4f}"

    csv_output_data.append(record)

  if csv_output_data:
    results_df = pd.DataFrame(csv_output_data)

    # 根據閾值生成文件名
    # actual_csv_filename = output_csv_filename.format(str(prediction_threshold).replace('.', '_'))
    results_df.to_csv(actual_output_csv_filename, header=True,index=False, encoding='utf-8-sig')
    print(f"=> 詳細預測結果 (閾值 {prediction_threshold}) 已保存至 {actual_output_csv_filename}")

  return current_metrics

# --- 步驟 5: 測試不同的閾值，輸出和儲存結果 ---
thresholds_to_test = [0.4, 0.5, 0.6] # 測試的閾值
all_experiment_metrics = {} # 存儲每個閾值的指標結果

# 當前的 id_list 是從 outside_data_json 中提取的 id，"test_texts_Re_jieba.json" 裡沒有id 欄位，所以這裡需要手動生成一個 id_list
# id_list = [f"sample_{i}" for i in range(len(test_dataset))] # Create a list of strings like "sample_0", "sample_1", etc.

for thresh in tqdm(thresholds_to_test, desc="Evaluating thresholds"):
  csv_filename = os.path.join(output_base_path,f"predictions_thresh_{str(thresh).replace('.', '_')}.csv") # 為每個閾值生成不同的CSV文件名
  # print(f"\n{csv_filename}")
  metrics_for_this_thresh = evaluate_and_save_with_threshold(
      id_list=id_list,
      probs_all_np=all_test_probs_np,
      true_labels_all_binarized_np=all_test_true_labels_np,
      original_texts_list=test_dataset.texts, # 從 dataset 實例獲取原始文本
      mlb_instance=mlb,
      prediction_threshold=thresh,
      actual_output_csv_filename=csv_filename,
      top_k_count=3
  )
  all_experiment_metrics[f"threshold_{thresh}"] = metrics_for_this_thresh

  # print("\n--- 所有閾值的評估指標總結 ---")
  # print(json.dumps(all_experiment_metrics, indent=2))

    # 你可以將 all_experiment_metrics 保存到一個 JSON 文件
with open(metrics_path, "w", encoding="utf-8") as f_summary:
  json.dump(all_experiment_metrics, f_summary, ensure_ascii=False, indent=4)
print("\n所有閾值的指標摘要已保存到", metrics_path)

