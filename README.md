# Classification-of-information-requests-from-Members

安裝套件：
pandas: 2.2.3
numpy: 2.1.3
torch: 2.7.0+cu128
transformers: 4.51.3
sklearn((scikit-learn)): 1.6.1
tqdm: 4.67.1
openpyxl: 3.1.5
tokenizers: 0.21.1
tensorflow: 2.19.0
keras: 3.9.2
regex: 2024.11.6
accelerate: 1.7.0
wandb: 0.19.11
matplotlib: 3.10.3
tensorboard: 2.19.0
jieba: 0.42.1
csv: 內建
os: 內建

```mermaid
---
title: Label processing flow
---
flowchart LR
    raw-data("raw data") --> row-merge["同案件編號/索資題目的進行合併"]
    row-merge --> nickname-check["nickname check
    如果在簡稱的對照中，進行名稱替換"]
    nickname-check --> vaild-check@{ label: "標籤合法性驗證(白名單匹配)\n    # 輸入標籤是否包含白名單一級/二級機關?\n    # 例如：輸入'台北市政府文化局'，白名單含'文化局'" }
    vaild-check -- 匹配成功 --> keep-secondary-check{"檢查是否為需要到二級的局處"}
    vaild-check -- 匹配失敗 --> error-check["資料捨棄"]
    keep-secondary-check -- Yes --> keep-secondary["Y-保留二級名稱"]
    keep-secondary-check -- NO --> not-keep-secondary["N-改變為一級單位名稱"]
    keep-secondary --> merge["合併一級、二級名稱"]
    merge --> final["最終標籤名稱"]
    not-keep-secondary --> final
    final --> is-over-limit{"判斷資料中是否超過10個以上的分辦機關"}
    is-over-limit -- YES --> over-limit["Y-變更標籤為其他"]
    is-over-limit -- NO --> not-over-limit["N-維持原標籤"]
```