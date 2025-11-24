# ⚠️ Streamlit Cloud 儲存機制說明

## 🔴 重要事實

### Streamlit Cloud 沒有持久化儲存！

```python
LOGS_DIR = Path("logs")  # 這在 Streamlit Cloud 是臨時的！
```

**當你看到「✅ 記錄已儲存至後端」時：**
- ✅ 檔案確實被寫入了
- ❌ 但存在臨時檔案系統
- 💀 App 重啟後會消失

**檔案位置**：
- 本地：`/home/linchiyi/BCS_AI_Chatbot_with_Evaluation/logs/` → 永久保存 ✅
- Streamlit Cloud：`/mount/src/repo/logs/` → **臨時！重啟即消失** ❌

## 💡 解決方案：必須上傳到 Google Drive

### 為什麼需要 Google Drive

| 方案 | 本地開發 | Streamlit Cloud | 持久化 |
|------|----------|-----------------|--------|
| `logs/` 資料夾 | ✅ 可用 | ❌ 臨時 | ❌ Cloud 會清空 |
| Google Drive | ✅ 永久 | ✅ 永久 | ✅ 完全持久化 |

### 目前狀況檢查

執行以下步驟來診斷：

```bash
# 1. 本地測試（確認程式碼沒問題）
cd ~/BCS_AI_Chatbot_with_Evaluation
streamlit run test_drive_quick.py

# 應該看到：
# ✅ DRIVE_FOLDER_ID
# ✅ oauth_token 存在
# ✅ Drive service 建立成功
# ✅ 上傳成功
```

如果本地測試通過，那問題出在 Streamlit Cloud 設定。

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

### 2. 檢查 Streamlit Cloud Logs

在 Streamlit Cloud 的 Manage app > Logs 中尋找：

**成功的訊息**：
```
✅ Token 已從 Streamlit Secrets 讀取
✅ Google Drive service 初始化成功
✅ 檔案已上傳：session_20251124_123456.json
```

**失敗的訊息**：
```
⚠️ 從 Streamlit Secrets 讀取 token 失敗
⚠️ Google Drive service 初始化失敗
❌ 上傳失敗
```

### 3. 常見問題

#### 問題 1：Token 已過期
**症狀**：本地可以，Cloud 不行  
**原因**：你本地的 token 是新的，但 Secrets 中的 token 是舊的  
**解決**：
```bash
python token_to_secrets.py  # 重新生成
# 複製新的 secrets.toml 到 Streamlit Cloud
```

#### 問題 2：Scopes 格式錯誤
**症狀**：Secrets 讀取失敗  
**原因**：TOML 格式問題  
**解決**：確保 scopes 用方括號 `[]` 包住

#### 問題 3：缺少欄位
**症狀**：Drive service 建立失敗  
**原因**：少了 client_secret 或其他欄位  
**解決**：檢查 token_info 的所有必要欄位

## 📝 Debug 流程

### Step 1：本地測試
```bash
streamlit run test_drive_quick.py
```
點擊「🚀 測試連線」和「📤 測試上傳」

### Step 2：部署 test_drive_quick.py 到 Cloud
在 Streamlit Cloud 建立一個測試 app：
- Repository: 同一個
- Branch: main
- Main file: `test_drive_quick.py`

這會顯示 Secrets 是否正確設定

### Step 3：檢查主 App
如果測試 app 通過，主 app (`app_emotion_guided.py`) 應該也會正常運作

## 🎯 最終確認

**本地開發**：
- `logs/` 資料夾：✅ 永久保存
- Google Drive：✅ 永久保存

**Streamlit Cloud**：
- `logs/` 資料夾：❌ 臨時，會清空
- Google Drive：✅ 永久保存（**唯一選擇**）

## 💡 建議

1. **本地開發**：兩個都有（方便本地查看）
2. **Cloud 部署**：只依賴 Google Drive
3. **不要依賴** Streamlit Cloud 的檔案系統來保存資料

---

**需要協助嗎？**

1. 執行 `streamlit run test_drive_quick.py`
2. 截圖結果
3. 告訴我看到什麼訊息
