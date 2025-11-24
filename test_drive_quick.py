"""
簡單的 Google Drive 連線診斷工具
用於快速檢查 Streamlit Cloud 上的 Drive 整合
"""
import streamlit as st

st.title("🔍 Google Drive 快速診斷")

st.write("## 檢查環境")

# 檢查是否在 Streamlit Cloud
import os
st.write(f"- 當前工作目錄: `{os.getcwd()}`")
st.write(f"- Python 路徑: `{os.path.dirname(os.__file__)}`")

# 檢查 Secrets
st.write("## 檢查 Secrets")
try:
    if 'DRIVE_FOLDER_ID' in st.secrets:
        st.success(f"✅ DRIVE_FOLDER_ID: {st.secrets['DRIVE_FOLDER_ID']}")
    else:
        st.error("❌ 找不到 DRIVE_FOLDER_ID")
    
    if 'oauth_token' in st.secrets:
        st.success("✅ oauth_token 存在")
        oauth_keys = list(st.secrets['oauth_token'].keys())
        st.write(f"  Keys: {oauth_keys}")
        
        # 檢查必要的 keys
        required = ['token', 'refresh_token', 'client_id', 'client_secret']
        for key in required:
            if key in st.secrets['oauth_token']:
                value = str(st.secrets['oauth_token'][key])
                st.write(f"  ✅ {key}: `{value[:30]}...`")
            else:
                st.error(f"  ❌ 缺少 {key}")
    else:
        st.error("❌ 找不到 oauth_token")
except Exception as e:
    st.error(f"檢查 Secrets 失敗: {e}")

# 測試 Google Drive
st.write("## 測試 Google Drive 連線")

if st.button("🚀 測試連線"):
    with st.spinner("正在測試..."):
        try:
            from google_drive_utils import get_drive_service
            
            # 顯示偵測過程
            st.write("### 初始化過程:")
            
            service = get_drive_service()
            
            if service:
                st.success("✅ Drive service 建立成功！")
                
                # 嘗試列出檔案
                st.write("### 測試 API 呼叫:")
                try:
                    results = service.files().list(
                        pageSize=5,
                        fields="files(id, name, createdTime)"
                    ).execute()
                    
                    items = results.get('files', [])
                    
                    if items:
                        st.write(f"找到 {len(items)} 個檔案:")
                        for item in items:
                            st.write(f"- {item['name']} (ID: {item['id'][:20]}...)")
                    else:
                        st.info("Drive 中沒有檔案（這可能是正常的）")
                    
                    st.success("🎉 Google Drive 連線完全正常！")
                    
                except Exception as e:
                    st.error(f"列出檔案時失敗: {e}")
                    st.exception(e)
            else:
                st.error("❌ Drive service 建立失敗")
                st.write("可能原因:")
                st.write("- Token 已過期")
                st.write("- Secrets 格式不正確")
                st.write("- 缺少必要的欄位")
                
        except Exception as e:
            st.error(f"測試失敗: {e}")
            st.exception(e)

# 測試上傳
st.write("## 測試檔案上傳")

if st.button("📤 測試上傳"):
    with st.spinner("正在上傳測試檔案..."):
        try:
            from pathlib import Path
            from google_drive_utils import get_drive_service, upload_to_drive
            import json
            from datetime import datetime
            
            # 建立測試檔案
            test_data = {
                "test": "這是測試檔案",
                "timestamp": datetime.now().isoformat(),
                "source": "Streamlit Cloud 診斷工具"
            }
            
            test_file = Path("test_upload.json")
            with test_file.open("w", encoding="utf-8") as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            
            st.write(f"✅ 測試檔案已建立: {test_file}")
            
            # 取得 service
            service = get_drive_service()
            
            if service:
                # 上傳
                folder_id = st.secrets.get('DRIVE_FOLDER_ID')
                
                file_id = upload_to_drive(
                    service=service,
                    file_path=test_file,
                    folder_id=folder_id,
                    mime_type='application/json'
                )
                
                if file_id:
                    st.success(f"🎉 上傳成功！File ID: {file_id}")
                    st.write(f"請到 Google Drive 資料夾檢查: {folder_id}")
                else:
                    st.error("❌ 上傳失敗（沒有返回 file_id）")
            else:
                st.error("❌ 無法取得 Drive service")
                
        except Exception as e:
            st.error(f"上傳測試失敗: {e}")
            st.exception(e)

st.write("---")
st.write("### 💡 如果測試失敗")
st.write("1. 檢查 Streamlit Cloud Secrets 中的 oauth_token 是否完整")
st.write("2. Token 可能已過期，需要重新執行 `python token_to_secrets.py`")
st.write("3. 檢查 Drive 資料夾 ID 是否正確")
