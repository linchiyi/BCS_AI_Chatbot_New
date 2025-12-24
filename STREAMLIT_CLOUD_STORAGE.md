## 🔧 Streamlit Cloud 部署檢查清單

### 1. Secrets 格式確認

你的 Secrets 應該**完全**是這個格式（沒有多餘的空行或註解）：

```toml
OPENAI_API_KEY = ""
DRIVE_FOLDER_ID = ""

[oauth_token]
token = ""
refresh_token = ""
token_uri = ""
client_id = ""
client_secret = ""
scopes = [
  "https://www.googleapis.com/auth/drive.file",
]
```

### 常見問題

#### 問題 1：Token 已過期
**症狀**：本地可以，Cloud 不行  
**原因**：你本地的 token 是新的，但 Secrets 中的 token 是舊的  
**解決**：
```bash
python token_to_secrets.py  # 重新生成
# 複製新的 secrets.toml 到 Streamlit Cloud
```



## Streamlit secerts setting
