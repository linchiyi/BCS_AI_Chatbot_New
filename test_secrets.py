"""
測試 Streamlit Secrets 是否正確設定
用於 debug Streamlit Cloud 部署問題
"""
import streamlit as st

st.title("🔍 Streamlit Secrets 檢查工具")

st.write("## 1. 檢查 Secrets 可用性")

try:
    # 檢查是否有 secrets
    if hasattr(st, 'secrets'):
        st.success("✅ st.secrets 可用")
        
        # 列出所有 secrets keys（不顯示值）
        st.write("### 已設定的 Secrets Keys:")
        secret_keys = list(st.secrets.keys())
        for key in secret_keys:
            if key == 'oauth_token':
                # 顯示 oauth_token 的子 keys
                oauth_keys = list(st.secrets['oauth_token'].keys())
                st.write(f"- `{key}`: {oauth_keys}")
            else:
                st.write(f"- `{key}`")
        
        # 檢查必要的 keys
        st.write("### 必要的 Secrets 檢查:")
        
        required_keys = {
            'OPENAI_API_KEY': 'OpenAI API Key',
            'DRIVE_FOLDER_ID': 'Google Drive 資料夾 ID',
            'oauth_token': 'OAuth Token'
        }
        
        all_good = True
        for key, description in required_keys.items():
            if key in st.secrets:
                st.success(f"✅ {description} (`{key}`) 已設定")
                
                # 顯示部分內容（前20字元）
                if key == 'oauth_token':
                    oauth_token = st.secrets['oauth_token']
                    required_oauth_keys = ['token', 'refresh_token', 'client_id', 'client_secret']
                    for oauth_key in required_oauth_keys:
                        if oauth_key in oauth_token:
                            value_preview = str(oauth_token[oauth_key])[:20] + "..."
                            st.write(f"  - `{oauth_key}`: {value_preview}")
                        else:
                            st.error(f"  ❌ 缺少 `oauth_token.{oauth_key}`")
                            all_good = False
                else:
                    value_preview = str(st.secrets[key])[:20] + "..."
                    st.write(f"  預覽: `{value_preview}`")
            else:
                st.error(f"❌ {description} (`{key}`) 未設定")
                all_good = False
        
        if all_good:
            st.success("🎉 所有必要的 Secrets 都已正確設定！")
        else:
            st.error("⚠️ 有部分 Secrets 缺失或不完整")
            
    else:
        st.error("❌ st.secrets 不可用（這不應該發生在 Streamlit 環境中）")
        
except Exception as e:
    st.error(f"❌ 檢查 Secrets 時發生錯誤：{e}")

st.write("---")
st.write("## 2. 測試 Google Drive 連線")

if st.button("測試 Google Drive 連線"):
    with st.spinner("正在測試..."):
        try:
            from google_drive_utils import get_drive_service
            
            service = get_drive_service()
            
            if service:
                st.success("✅ Google Drive service 初始化成功！")
                
                # 嘗試列出檔案
                try:
                    results = service.files().list(pageSize=5, fields="files(id, name)").execute()
                    items = results.get('files', [])
                    
                    if items:
                        st.write("### 你的 Drive 中的檔案（前 5 個）：")
                        for item in items:
                            st.write(f"- {item['name']}")
                    else:
                        st.info("Drive 中沒有找到檔案（這可能是正常的）")
                        
                    st.success("🎉 Google Drive 連線測試成功！")
                except Exception as e:
                    st.error(f"❌ 列出檔案時發生錯誤：{e}")
            else:
                st.error("❌ Google Drive service 初始化失敗")
                st.write("請檢查：")
                st.write("- Secrets 中的 oauth_token 是否完整")
                st.write("- Token 是否已過期（需要重新生成）")
                
        except Exception as e:
            st.error(f"❌ 測試過程發生錯誤：{e}")
            st.exception(e)

st.write("---")
st.write("## 3. 環境資訊")

import sys
import os

st.write(f"- Python 版本: {sys.version}")
st.write(f"- Streamlit 版本: {st.__version__}")
st.write(f"- 當前工作目錄: {os.getcwd()}")

# 檢查檔案系統
st.write("### 檔案系統檢查:")
from pathlib import Path

files_to_check = [
    'credentials.json',
    'token.pickle',
    'google_drive_utils.py',
    'session_logger.py',
]

for filename in files_to_check:
    if Path(filename).exists():
        st.write(f"✅ `{filename}` 存在")
    else:
        st.write(f"❌ `{filename}` 不存在")
