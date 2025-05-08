import os
import csv
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    TrainingArguments, Trainer, TrainerCallback, EvalPrediction
)

# ===== 資料讀取與預處理 =====

df = pd.read_excel("113索資.xlsx")[['索取資料題目', '承辦機關']].dropna()
df.columns = ['text', 'label']

valid_labels = {
    "秘書處", "民政局", "財政局", "教育局", "產業發展局", "工務局", "交通局", "社會局", "勞動局",
    "警察局", "衛生局", "環境保護局", "都市發展局", "文化局", "消防局", "捷運工程局", "臺北翡翠水庫管理局",
    "觀光傳播局", "地政局", "兵役局", "體育局", "資訊局", "法務局", "青年局", "主計處", "人事處", "政風處",
    "公務人員訓練處", "研究發展考核委員會", "都市計畫委員會", "原住民族事務委員會", "客家事務委員會",
    "臺北自來水事業處", "臺北大眾捷運股份有限公司",
    "士林區公所", "大同區公所", "大安區公所", "中山區公所", "中正區公所", "內湖區公所", "文山區公所",
    "北投區公所", "松山區公所", "信義區公所", "南港區公所", "萬華區公所", "工務局大地工程處", "工務局公園路燈工程管理處",
    "工務局水利工程處", "工務局新建工程處", "工務局衛生下水道工程處", "公務人員訓練處", "孔廟管理委員會", "建築管理工程處",
    "家庭暴力暨性侵害防治中心", "交通管制工程處", "市場處", "研究發展考核委員會", "動物保護處", "停車管理工程處", "警察局交通警察大隊",
    "士林區戶政事務所", "大安地政事務所", "公共運輸處", "文山區戶政事務所", "古亭地政事務所", "市政大樓公共事務管理中心",
    "台北市立天文科學教育館", "台北市立文獻館", "台北市立交響樂團", "台北市立美術館","台北市立浩然敬老院",
    "台北市立動物園", "台北市立陽明教養院", "台北市立圖書館", "地政局土地開發總隊", "松山地政事務所", "青少年發展暨家庭教育中心",
    "信義區戶政事務所", "社會局建成地政事務所", "商業處", "捷運工程局第一區工程處", "捷運工程局第二區工程處",
    "捷運工程局機電系統工程處","都市更新處", "稅捐稽徵處", "環境保護局環保稽查大隊", "殯葬管理處","職能發展學院", "藝文推廣處" ,
    "警察局刑事警察大隊"
}

def normalize_labels(label_list):
    return list({label if label in valid_labels else "其他" for label in label_list})

grouped = df.groupby('text')['label'].apply(lambda x: normalize_labels(list(set(x)))).reset_index()
grouped.columns = ['text', 'labels']
grouped.to_csv("grouped_text_labels.csv", index=False, encoding="utf-8-sig")

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(grouped['labels'])

train_texts, val_texts, train_labels, val_labels = train_test_split(
    grouped['text'], y, test_size=0.2, random_state=42
)

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

# ===== Dataset 定義 =====

class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            list(tqdm(texts, desc="Tokenizing")),
            truncation=True, padding=True, max_length=512
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = MultiLabelDataset(train_texts, train_labels)
val_dataset = MultiLabelDataset(val_texts, val_labels)

# ===== 模型載入 =====

model = BertForSequenceClassification.from_pretrained(
    "hfl/chinese-roberta-wwm-ext",
    num_labels=y.shape[1],
    problem_type="multi_label_classification"
)

# ===== 評估指標 =====

def compute_metrics(pred: EvalPrediction):
    logits, labels = pred
    preds = torch.sigmoid(torch.tensor(logits)).numpy()
    labels = labels if isinstance(labels, np.ndarray) else labels.numpy()
    labels = labels.astype(int)
    preds_binary = (preds >= 0.5).astype(int)

    micro_f1 = f1_score(labels, preds_binary, average='micro')
    macro_f1 = f1_score(labels, preds_binary, average='macro')
    subset_accuracy = (preds_binary == labels).all(axis=1).mean()
    sample_accuracy = (preds_binary & labels).sum(axis=1) / (
        (preds_binary | labels).sum(axis=1) + 1e-8
    )
    sample_accuracy = sample_accuracy.mean()

    return {
        "micro_f1": round(micro_f1, 6),
        "macro_f1": round(macro_f1, 6),
        "subset_accuracy": round(subset_accuracy, 6),
        "sample_accuracy": round(sample_accuracy, 6)
    }

# ===== 自訂進度條 Callback =====

class TQDMProgressBar(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.train_bar = tqdm(total=state.max_steps, desc="Training")

    def on_step_end(self, args, state, control, **kwargs):
        self.train_bar.update(1)
        if 'logs' in kwargs and 'loss' in kwargs['logs']:
            self.train_bar.set_postfix(loss=kwargs["logs"]["loss"])

    def on_train_end(self, args, state, control, **kwargs):
        self.train_bar.close()

# ===== 訓練參數設定 =====

output_dir = "./model_output10"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="micro_f1",
    greater_is_better=True,
    logging_dir=None,
    report_to=[]
)

# ===== Trainer 初始化與訓練 =====

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[TQDMProgressBar]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("開始訓練...")
trainer.train()

# ===== 使用最佳模型進行預測並儲存 CSV =====

print("使用最佳模型進行評估與預測...")
predictions = trainer.predict(val_dataset)
logits = predictions.predictions
labels = predictions.label_ids

probs = torch.sigmoid(torch.tensor(logits)).numpy()
preds_binary = (probs >= 0.5).astype(int)
val_text_list = list(val_texts)  # 確保順序正確

output_path = "./predictions10.csv"
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_MINIMAL)
    header = ["索取資料題目", "預測承辦機關", "實際承辦機關"] + list(mlb.classes_)
    writer.writerow(header)

    for text, prob, true_labels in zip(val_text_list, probs, labels):
        predicted_labels = [mlb.classes_[i] for i, p in enumerate(prob) if p >= 0.5]
        true_label_names = [mlb.classes_[i] for i in range(len(true_labels)) if true_labels[i] == 1]
        if not predicted_labels:
            predicted_labels = ["其他"]
        if not true_label_names:
            true_label_names = ["其他"]

        writer.writerow([text, ";".join(predicted_labels), ";".join(true_label_names)] + list(prob))

print(f"預測結果已儲存至 {output_path}")
