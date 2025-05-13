import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from dotenv import load_dotenv
import torch
from transformers import TextClassificationPipeline
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    TrainingArguments, Trainer, TrainerCallback, EvalPrediction
)

# apply .env environment variables
load_dotenv()

# label mapping
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

# Convert the set to a list for indexing
valid_labels_list = list(valid_labels)

def normalize_labels(label_list):
    return list({label if label in valid_labels else "其他" for label in label_list})

tarining_set_path = os.getenv("TRAINING_SET_PATH")
if tarining_set_path:
    excel_file_path = f"{tarining_set_path}/113索資.xlsx"
else:
    excel_file_path = f"113索資.xlsx"
df = pd.read_excel(excel_file_path)[['索取資料題目', '承辦機關']].dropna()
df.columns = ['text', 'label']

grouped = df.groupby('text')['label'].apply(lambda x: normalize_labels(list(set(x)))).reset_index()
grouped.columns = ['text', 'labels']
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(grouped['labels'])

train_texts, val_texts, train_labels, val_labels = train_test_split(
    grouped['text'], y, test_size=0.2, random_state=42
)

# load pretrained model and tokenizer
persist_model_path = "./model_saved"
model = BertForSequenceClassification.from_pretrained(f"{persist_model_path}")
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

while True:
    # request a console input the replace the hardcoded text
    print("請輸入要分類的索取資料題目:")
    user_input = input().strip()

    if not user_input:
        print("請輸入有效的索取資料題目")
        exit(1)

    predict_result = pipe(user_input)
    # find the top 3 scores in predict_result[0]
    top_three_score_result = sorted(predict_result[0], key=lambda x: x['score'], reverse=True)[:3]
    # convert the label to the original valid_labels
    top_three_score_result = [
        {
            "label": label['label'],
            "score": label['score'],
            "label_name": mlb.classes_[int(label['label'].replace("LABEL_", ""))] if label['label'].startswith("LABEL_") else label['label']
        }
        for label in top_three_score_result
    ]

    # print the top 3 scores and label_name
    for label in top_three_score_result:
        print(f"Label: {label['label_name']}, Score: {label['score']:.4f}")

# print(top_three_score_result)