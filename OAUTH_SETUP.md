# Google Drive OAuth 2.0 設定指南

## 問題說明

Service Account 無法直接上傳到個人 Google Drive（錯誤：`storageQuotaExceeded`）。

解決方案：改用 **OAuth 2.0 使用者授權**

## 設定步驟

### 1. 建立 OAuth 2.0 憑證

1. 前往 [Google Cloud Console](https://console.cloud.google.com/)
2. 選擇你的專案（或建立新專案）
3. 左側選單：**API 和服務** → **憑證**
4. 點擊 **建立憑證** → **OAuth 用戶端 ID**
5. 應用程式類型選擇：**電腦應用程式**
6. 名稱隨意填寫（例如：`OSCE Chatbot Desktop`）
7. 點擊 **建立**

### 2. 下載憑證檔案

1. 在憑證列表中找到剛建立的 OAuth 用戶端
2. 點擊右側的 **下載 JSON** 圖示（⬇️）
3. 將下載的檔案重新命名為 `credentials.json`
4. 放到專案根目錄：
   ```
   BCS_AI_Chatbot_with_Evaluation/
   ├── credentials.json  ← 放這裡
   ├── app_emotion_guided.py
   ├── session_logger.py
   └── ...
   ```

### 3. 啟用 Google Drive API

1. 在 Google Cloud Console 左側選單
2. **API 和服務** → **已啟用的 API 和服務**
3. 點擊 **啟用 API 和服務**
4. 搜尋 "Google Drive API"
5. 點擊進入後選擇 **啟用**

### 4. 測試連線

執行測試腳本：
```bash
cd ~/BCS_AI_Chatbot_with_Evaluation
python google_drive_utils.py
```

**第一次執行時**：
- 會自動開啟瀏覽器
- 要求你登入 Google 帳號並授權
- 授權後會自動儲存 token 到 `token.pickle`
- 之後就不需要再登入了

**預期輸出**：
```
✅ 授權成功！
✅ Token 已儲存到 token.pickle
=== 測試 Google Drive 連線 ===

你的 Drive 中的檔案（前 5 個）：
  - 檔案1.pdf (1ABC...xyz)
  - 檔案2.docx (1DEF...uvw)
  ...

✅ Google Drive 連線測試成功！
```

### 5. 執行完整測試

```bash
python test_session_logger.py
```

應該會看到：
```
=== 測試本地 logging ===
✅ 本地記錄成功

=== 測試 Google Drive 上傳 ===
✅ Google Drive service 初始化成功
✅ 檔案已上傳：session_XXXXXX.json
   Drive ID: 1ABC...
   連結: https://drive.google.com/...
```

## 檔案說明

### `credentials.json` (需要保密)
- OAuth 2.0 用戶端憑證
- 從 Google Cloud Console 下載
- **不要上傳到 Git**（已加入 .gitignore）

### `token.pickle` (自動生成)
- 授權 token 快取檔案
- 首次授權後自動建立
- 避免每次都要重新登入
- **不要上傳到 Git**（已加入 .gitignore）

## 常見問題

### Q: 為什麼不能用 Service Account？
A: Service Account 沒有自己的儲存空間配額，只能上傳到：
   - Google Workspace 的共用雲端硬碟
   - 使用 domain-wide delegation

   個人 Google Drive 必須用 OAuth 2.0

### Q: Token 會過期嗎？
A: Token 有效期通常是 7 天，但程式會自動更新，你不需要手動處理

### Q: 如何重新授權？
A: 刪除 `token.pickle` 後重新執行程式即可

### Q: 授權後可以在 Streamlit app 中使用嗎？
A: 可以！首次在終端機執行測試並完成授權後，Streamlit app 會自動使用儲存的 token

## 安全性注意事項

✅ **已加入 .gitignore 的檔案**：
- `credentials.json` - OAuth 憑證
- `token.pickle` - 授權 token
- `service_account.json` - Service Account 憑證（如果有）

⚠️ **千萬不要**：
- 上傳這些檔案到 GitHub
- 分享給其他人
- 公開在任何地方

## 與 Service Account 的差異

| 特性 | Service Account | OAuth 2.0 |
|------|----------------|-----------|
| 適用對象 | Google Workspace 組織 | 個人 Google Drive |
| 授權方式 | JSON 金鑰檔案 | 瀏覽器登入授權 |
| 首次設定 | 僅需放置 JSON | 需要瀏覽器授權 |
| 後續使用 | 自動 | 自動（token 快取） |
| 上傳位置 | 共用雲端硬碟 | 個人 Drive |

## 下一步

完成設定後：
1. 確認測試都通過
2. 執行 Streamlit app：`streamlit run app_emotion_guided.py`
3. 完成一次完整對話並評分
4. 檢查 Drive 資料夾是否有上傳成功

---

*最後更新：2025-01-13*
