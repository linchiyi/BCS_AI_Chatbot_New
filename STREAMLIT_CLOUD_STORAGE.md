## 🔧 Streamlit Cloud 部署檢查清單

### 1. Secrets 格式確認

你的 Secrets 應該**完全**是這個格式（沒有多餘的空行或註解）：

```toml
OPENAI_API_KEY = "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"
DRIVE_FOLDER_ID = "16HRRkutsZcscFkk4Q7XgJPEjbz3nurod"

[oauth_token]
token = "ya29.a0ATi6K2vhTyHgmtFA7-z0-n1Nm0HUsZS4crsb7oAJTAJWQLc7NbxPeM0eTZgzvl5gb5cb5jiAY-GCSI2zDkM70nEnVDwubB-rClEdyankW3_o9roGlfDKAC5moCaFXAXqJn7aM1FyhufLj5fHnJQ2sp7URN5J2-KHZcmFvFRkyHZY_LFMnfYau7Dk0I5BjJRxRHjQnRPBaCgYKAdgSARMSFQHGX2MiOdX7zBMO3ODJgUTlrOPpNg0207"
refresh_token = "1//0e7A8N0_THZh0CgYIARAAGA4SNwF-L9Ir0bRGI7w40qtNRUvRdar2UJ2GLuVSelRuXhLxVG242N2ERJqLRcnAWY3ve7UU1731N9k"
token_uri = "https://oauth2.googleapis.com/token"
client_id = "721534481068-ia7tmg6es7oqhl08l3klpgboqqahj0q6.apps.googleusercontent.com"
client_secret = "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"
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

williamyuan.tw@gmail.com
extraboy25@gmail.com
linchiyi.ii14@nycu.edu.tw
linqiyi0071@gmail.com


## Streamlit secerts setting
OPENAI_API_KEY = "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"
DRIVE_FOLDER_ID = "16HRRkutsZcscFkk4Q7XgJPEjbz3nurod"

[oauth_token]
token = "ya29.a0ATi6K2vhTyHgmtFA7-z0-n1Nm0HUsZS4crsb7oAJTAJWQLc7NbxPeM0eTZgzvl5gb5cb5jiAY-GCSI2zDkM70nEnVDwubB-rClEdyankW3_o9roGlfDKAC5moCaFXAXqJn7aM1FyhufLj5fHnJQ2sp7URN5J2-KHZcmFvFRkyHZY_LFMnfYau7Dk0I5BjJRxRHjQnRPBaCgYKAdgSARMSFQHGX2MiOdX7zBMO3ODJgUTlrOPpNg0207"
refresh_token = "1//0e7A8N0_THZh0CgYIARAAGA4SNwF-L9Ir0bRGI7w40qtNRUvRdar2UJ2GLuVSelRuXhLxVG242N2ERJqLRcnAWY3ve7UU1731N9k"
token_uri = "https://oauth2.googleapis.com/token"
client_id = "721534481068-ia7tmg6es7oqhl08l3klpgboqqahj0q6.apps.googleusercontent.com"
client_secret = "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"
scopes = [
  "https://www.googleapis.com/auth/drive.file",
]