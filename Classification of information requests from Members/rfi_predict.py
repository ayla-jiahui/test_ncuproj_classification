import os
import dotenv
from transformers import TextClassificationPipeline
import json
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    TrainingArguments, Trainer, TrainerCallback, EvalPrediction
    )
from dotenv import load_dotenv

load_dotenv()

base_path = os.getcwd() + f"/{os.getenv('BASE_PATH', 'Classification of information requests from Members')}"
print(f"base_path: {base_path}")

# 讀取 mlb_classes JSON檔
with open(f"{base_path}/{os.getenv('MLB_PATH', 'model_saved/20250523')}/mlb_classes.json", "r", encoding="utf-8") as f:
    classes_list = json.load(f)

# 建立新的 MultiLabelBinarizer 並指定 classes
mlb = MultiLabelBinarizer(classes=classes_list)
mlb.classes_ = classes_list  # 明確指定 classes_ 屬性
print(mlb.classes_)
# print(len(mlb.classes_))

# load pretrained model and tokenizer
persist_model_path = f"{base_path}/{os.getenv('MODEL_PRESIST_PATH', 'model_saved/20250523/model_saved_0523')}"
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