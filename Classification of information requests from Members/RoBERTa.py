import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import (
    BertForSequenceClassification,TrainingArguments, 
    Trainer, TrainerCallback, EvalPrediction, AutoTokenizer)
import torch.nn.functional as F
import matplotlib.pyplot as plt
import jieba
import re

# Focal Loss 
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma 
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

# Trainer 
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss()
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.focal_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ===== 資料讀取與預處理 =====
df1 = pd.read_excel("113索資.xlsx")[['索取資料題目', '承辦機關']]
df2 = pd.read_excel("114索資.xlsx")[['索取資料題目', '承辦機關']]
df = pd.concat([df1, df2]).dropna()
df.columns = ['text', 'label']
df['text'] = df['text'].astype(str).str.strip()
df['label'] = df['label'].astype(str).str.strip()
df = df[(df['text'] != '') & (df['label'] != '')]

# ===== 移除贅詞 =====
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

def remove_redundant_words(text):
    for word in redundant_words:
        text = text.replace(word, "")
    return text.strip()

df['text'] = df['text'].apply(remove_redundant_words)

# ===== 標籤對應與正規化 =====
label_mapping = {
"市政大樓公共事務管理中心": "秘書處",

    "殯葬管理處": "民政局", 
    "孔廟管理委員會": "民政局",
    
    "稅捐稽徵處": "財政局", 
    "動產質借處": "財政局",
    
    "圖書館": "教育局", 
    "動物園": "教育局", 
    "教師研習中心": "教育局", 
    "教研中心":"教育局",
    "天文科學教育館": "教育局", 
    "家庭教育中心": "教育局",
    "青少年發展暨家庭教育中心": "教育局",

    "停車管理工程處": "交通局", 
    "交通管制工程處": "交通局", 
    "公共運輸處": "交通局", 
    "交通事件裁決所": "交通局",

    "陽明教養院": "社會局", 
    "浩然敬老院": "社會局", 
    "家庭暴力暨性侵害防治中心": "社會局",

    "勞動檢查處": "勞動局", 
    "就業服務處": "勞動局", 
    "勞動力重建運用處": "勞動局", 
    "職能發展學院": "勞動局",

    "警察局保安警察大隊": "警察局", 
    "警察局刑事警察大隊": "警察局", 
    "警察局交通警察大隊": "警察局", 
    "警察局少年警察隊": "警察局",
    "警察局婦幼警察隊": "警察局", 
    "警察局捷運警察隊": "警察局", 
    "警察局通信隊": "警察局",
    "警察局大同分局": "警察局",
    "警察局萬華分局": "警察局",
    "警察局中山分局": "警察局",
    "警察局大安分局": "警察局",
    "警察局中正第一分局": "警察局",
    "警察局中正第二分局": "警察局",
    "警察局松山分局": "警察局",
    "警察局信義分局": "警察局",
    "警察局士林分局": "警察局",
    "警察局北投分局": "警察局",
    "警察局文山第一分局": "警察局",
    "警察局文山第二分局": "警察局",
    "警察局南港分局": "警察局",
    "警察局內湖分局": "警察局",

    "聯合醫院": "衛生局",

    "環境保護局環保稽查大隊": "環境保護局", 
    "環境保護局內湖垃圾焚化廠": "環境保護局", 
    "環境保護局木柵垃圾焚化廠": "環境保護局", 
    "環境保護局北投垃圾焚化廠": "環境保護局",

    "國樂團": "文化局", 
    "交響樂團": "文化局", 
    "美術館": "文化局", 
    "中山堂管理所": "文化局", 
    "文獻館": "文化局", 
    "藝文推廣處": "文化局",

    "捷運工程局第一區工程處": "捷運工程局", 
    "捷運工程局第二區工程處": "捷運工程局", 
    "捷運工程局機電系統工程處": "捷運工程局",

    "臺北廣播電臺": "觀光傳播局",

    "地政局土地開發總隊": "地政局",

    "臺北自來水事業處工程總隊": "臺北自來水事業處",

    "松山區戶政事務所": "民政局",
    "信義區戶政事務所": "民政局",
    "大安區戶政事務所": "民政局",
    "中山區戶政事務所": "民政局",
    "中正區戶政事務所": "民政局",
    "大同區戶政事務所": "民政局",
    "南港區戶政事務所": "民政局",
    "內湖區戶政事務所": "民政局",
    "士林區戶政事務所": "民政局",
    "北投區戶政事務所": "民政局",
    "文山區戶政事務所": "民政局",
    "萬華區戶政事務所": "民政局",    

    "松山地政事務所": "地政局",
    "大安地政事務所": "地政局",
    "中山地政事務所": "地政局",
    "古亭地政事務所": "地政局",
    "士林地政事務所": "地政局",
    "建成地政事務所": "地政局",

    "松山區公所": "民政局",
    "信義區公所": "民政局",
    "大安區公所": "民政局",
    "中山區公所": "民政局",
    "中正區公所": "民政局",
    "大同區公所": "民政局",
    "南港區公所": "民政局",
    "內湖區公所": "民政局",
    "士林區公所": "民政局",
    "北投區公所": "民政局",
    "文山區公所": "民政局",
    "萬華區公所": "民政局",

}


valid_labels = {
    "臺北市政府",
    "秘書處", "民政局", "財政局", "教育局", "產業發展局", "工務局", "交通局", "社會局", "勞動局", 
    "警察局", "衛生局", "環境保護局", "都市發展局", "文化局", "消防局", "捷運工程局", "臺北翡翠水庫管理局", "觀光傳播局", 
    "地政局", "兵役局", "體育局", "資訊局", "法務局", "青年局", "主計處", "人事處", "政風處", 
    "公務人員訓練處", "研究發展考核委員會", "都市計畫委員會", "原住民族事務委員會", "客家事務委員會", "臺北自來水事業處", 
    "臺北大眾捷運股份有限公司", "工務局新建工程處", "工務局水利工程處", "工務局公園路燈工程管理處", 
    "工務局衛生下水道工程處", "工務局大地工程處", "市場處", "商業處", "動物保護處", "都市更新處", "建築管理工程處",
}

df['label'] = df['label'].replace(label_mapping)

def normalize_labels(label_list):
    return list({label if label in valid_labels else "其他" for label in label_list})

# 假設你的文字欄位名稱是 'text'
pattern = r'\b[A-Za-z]\d{5}-\d{8}\b'

# 用正則式替換掉符合格式的文字
df['text'] = df['text'].apply(lambda x: re.sub(pattern, '', str(x)))

grouped = df.groupby('text')['label'].apply(lambda x: normalize_labels(sorted(set(x)))).reset_index()
grouped.columns = ['text', 'labels']
grouped['label_count'] = grouped['labels'].apply(len)
grouped = grouped[grouped['label_count'] < 10].drop(columns='label_count')
grouped.to_csv("grouped_text_labels.csv", index=False, encoding="utf-8-sig")

# ===== 分割資料集（train/val/test） =====
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(grouped['labels'])

X_temp, X_test, y_temp, y_test = train_test_split(grouped['text'], y, test_size=0.2, random_state=42)
train_texts, val_texts, train_labels, val_labels = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42)

# ===== 斷句 =====
train_texts = train_texts.apply(lambda x: " ".join(jieba.cut(x)))
val_texts = val_texts.apply(lambda x: " ".join(jieba.cut(x)))
X_test = X_test.apply(lambda x: " ".join(jieba.cut(x)))

train_texts.to_csv("train_texts_Re_jieba.csv", index=False, encoding='utf-8-sig')
val_texts.to_csv("val_texts_Re_jieba.csv", index=False, encoding='utf-8-sig')
X_test.to_csv("test_texts_Re_jieba.csv", index=False, encoding='utf-8-sig')

# ===== 確認結果 =====
print('模式- 預設： ', ' | '.join(train_texts[:3])) 

# ===== 資料集定義 =====
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext", use_fast=True)

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
test_dataset = MultiLabelDataset(X_test, y_test)

# ===== 模型與訓練設定 =====
model = BertForSequenceClassification.from_pretrained(
    "hfl/chinese-roberta-wwm-ext",
    num_labels=y.shape[1],
    problem_type="multi_label_classification"
)
def compute_metrics(pred: EvalPrediction):
    logits = pred.predictions
    labels = pred.label_ids.astype(int)
    preds = torch.sigmoid(torch.tensor(logits)).numpy()
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

class TQDMProgressBar(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.train_bar = tqdm(total=state.max_steps, desc="Training")

    def on_step_end(self, args, state, control, **kwargs):
        self.train_bar.update(1)
        if 'logs' in kwargs and 'loss' in kwargs['logs']:
            self.train_bar.set_postfix(loss=kwargs["logs"]["loss"])

    def on_train_end(self, args, state, control, **kwargs):
        self.train_bar.close()

output_dir = "./model_regularization_jieba2"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=250,
    logging_strategy="epoch",
    logging_dir='./logs',
    learning_rate=5e-5,
    logging_steps=500,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    report_to=["wandb"],
    eval_steps=500
)

trainer = CustomTrainer(
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
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

#===== 預測與儲存函式 =====
def predict_and_save(texts, labels, filename, batch_size=32):
    model.eval()
    all_probs = []
    device = model.device

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Predicting {filename}"):
            batch_texts = texts[i:i+batch_size]
            batch_encodings = tokenizer(
                list(batch_texts),
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            batch_encodings = {k: v.to(device) for k, v in batch_encodings.items()}
            outputs = model(**batch_encodings)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            all_probs.extend(probs)

    probs = np.array(all_probs)
    preds_binary = (probs >= 0.5).astype(int)                            
    predicted_labels = mlb.inverse_transform(preds_binary)
    true_labels = mlb.inverse_transform(labels.astype(int))



    # ===== 顯示評估指標到 terminal =====
    micro_f1 = f1_score(labels, preds_binary, average="micro")
    macro_f1 = f1_score(labels, preds_binary, average="macro")
    subset_acc = (preds_binary == labels).all(axis=1).mean()
    sample_acc = (preds_binary & labels).sum(axis=1) / ((preds_binary | labels).sum(axis=1) + 1e-8)
    sample_acc = sample_acc.mean()
    micro_precision = precision_score(labels, preds_binary, average="micro", zero_division=0)
    micro_recall = recall_score(labels, preds_binary, average="micro", zero_division=0)

    print(f"\n=== 評估結果（{filename}）===")
    print(f"Micro F1:        {micro_f1:.6f}")
    print(f"Macro F1:        {macro_f1:.6f}")
    print(f"Subset Accuracy: {subset_acc:.6f}")
    print(f"Sample Accuracy: {sample_acc:.6f}")
    print(f"Micro_precison: {micro_precision:.6f}")
    print(f"Micro_recall: {micro_recall:.6f}")
    print("=" * 35 + "\n")

    # ===== 儲存結果到 CSV =====
    truncated_texts = [text[:200] + "..." if len(text) > 200 else text for text in texts]
    results_df = pd.DataFrame({ 
        "text": truncated_texts,
        "true_labels": [", ".join(lbls) for lbls in true_labels],
        "predicted_labels": [", ".join(lbls) for lbls in predicted_labels]
    })

    # 加入 sigmoid 數值
    label_names = mlb.classes_
    sigmoid_df = pd.DataFrame(probs, columns=label_names)
    results_df = pd.concat([results_df, sigmoid_df], axis=1)

    # 在第一列顯示標籤名稱
    results_df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"{filename} 已儲存。")

# ===== 儲存驗證集預測 =====
predict_and_save(val_texts, val_labels, "val_predictions_regularization_jieba2.csv")

# ===== 儲存測試集預測 =====
predict_and_save(X_test, y_test, "test_predictions_regularization_jieba.csv")

# ===== 訓練結束後畫的loss變化圖 =====
loss_log = trainer.state.log_history
loss_values = [entry['loss'] for entry in loss_log if 'loss' in entry]
epochs = list(range(1,len(loss_values)+1))

plt.figure(figsize=(10,5))
plt.plot(epochs, loss_values, label="Training Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve_regularization_jieba2.png")