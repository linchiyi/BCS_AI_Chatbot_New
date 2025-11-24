"""
Google Drive 工具模組
支援 OAuth 2.0 使用者授權（個人 Google Drive）
同時支援本地開發和 Streamlit Cloud 部署
"""
import os
import pickle
import json
import base64
from pathlib import Path
from typing import Optional

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False

# 檢查是否在 Streamlit 環境
try:
    import streamlit as st
    HAS_STREAMLIT = True
except:
    HAS_STREAMLIT = False

# Google Drive API 權限範圍
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def get_credentials_from_streamlit_secrets():
    """
    從 Streamlit Secrets 讀取 OAuth token
    
    Secrets 格式：
    [oauth_token]
    token = "xxx"
    refresh_token = "xxx"
    token_uri = "https://oauth2.googleapis.com/token"
    client_id = "xxx"
    client_secret = "xxx"
    scopes = ["https://www.googleapis.com/auth/drive.file"]
    
    Returns:
        Credentials 物件，如果失敗則返回 None
    """
    print("🔍 [DEBUG] 檢查 Streamlit Secrets...")
    
    if not HAS_STREAMLIT:
        print("⚠️ [DEBUG] Streamlit 模組不可用")
        return None
    
    try:
        if 'oauth_token' not in st.secrets:
            print("❌ [DEBUG] Secrets 中找不到 oauth_token")
            print(f"📋 [DEBUG] 可用的 Secrets keys: {list(st.secrets.keys())}")
            return None
        
        print("✅ [DEBUG] 找到 oauth_token")
        token_info = st.secrets['oauth_token']
        
        # 檢查必要欄位
        required_fields = ['token', 'refresh_token', 'client_id', 'client_secret']
        for field in required_fields:
            if field not in token_info:
                print(f"❌ [DEBUG] oauth_token 缺少欄位: {field}")
                return None
            else:
                print(f"✅ [DEBUG] oauth_token.{field} 存在")
        
        creds = Credentials(
            token=token_info.get('token'),
            refresh_token=token_info.get('refresh_token'),
            token_uri=token_info.get('token_uri', 'https://oauth2.googleapis.com/token'),
            client_id=token_info.get('client_id'),
            client_secret=token_info.get('client_secret'),
            scopes=token_info.get('scopes', SCOPES)
        )
        
        print(f"🔑 [DEBUG] Credentials 建立成功，expired={creds.expired}")
        
        # 如果 token 過期，嘗試更新
        if creds.expired and creds.refresh_token:
            print("⏳ [DEBUG] Token 已過期，嘗試更新...")
            try:
                creds.refresh(Request())
                print("✅ Token 已從 Streamlit Secrets 讀取並更新")
            except Exception as e:
                print(f"❌ Token 更新失敗：{e}")
                return None
        else:
            print("✅ Token 已從 Streamlit Secrets 讀取")
        
        return creds
        
    except Exception as e:
        print(f"❌ 從 Streamlit Secrets 讀取 token 失敗：{e}")
        import traceback
        traceback.print_exc()
        return None

def get_drive_service(credentials_file: str = 'credentials.json', token_file: str = 'token.pickle'):
    """
    初始化 Google Drive API service（使用 OAuth 2.0）
    
    優先順序：
    1. Streamlit Cloud: 從 st.secrets['oauth_token'] 讀取
    2. 本地: 從 token.pickle 讀取
    3. 本地: 需要瀏覽器授權（第一次）
    
    Args:
        credentials_file: OAuth 2.0 憑證檔案路徑（從 GCP Console 下載）
        token_file: Token 快取檔案路徑（自動生成）
        
    Returns:
        Google Drive API service 物件，如果失敗則返回 None
    """
    print("\n" + "="*60)
    print("🚀 [DEBUG] 開始初始化 Google Drive service")
    print("="*60)
    
    if not GOOGLE_DRIVE_AVAILABLE:
        print("❌ Google Drive API libraries 未安裝")
        return None
    
    creds = None
    
    # 方法 1: 從 Streamlit Secrets 讀取（Streamlit Cloud 環境）
    print(f"📍 [DEBUG] 方法 1: 檢查 Streamlit Secrets (HAS_STREAMLIT={HAS_STREAMLIT})")
    if HAS_STREAMLIT:
        creds = get_credentials_from_streamlit_secrets()
        if creds:
            print("🎯 [DEBUG] 使用 Streamlit Secrets 中的憑證")
            try:
                service = build('drive', 'v3', credentials=creds)
                print("✅ 使用 Streamlit Secrets 中的 OAuth token")
                print("="*60 + "\n")
                return service
            except Exception as e:
                print(f"❌ 使用 Secrets token 建立 service 失敗：{e}")
                creds = None
        else:
            print("⚠️ [DEBUG] Streamlit Secrets 中沒有有效的憑證")
    
    # 方法 2: 從本地檔案讀取 token
    print(f"📍 [DEBUG] 方法 2: 檢查本地 token.pickle ({token_file})")
    if os.path.exists(token_file):
        print(f"✅ [DEBUG] 找到 {token_file}")
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)
        print(f"✅ 從 {token_file} 讀取 token")
    else:
        print(f"⚠️ [DEBUG] 找不到 {token_file}")
    
    # 如果沒有有效的憑證，需要重新登入
    print(f"📍 [DEBUG] 檢查憑證有效性: creds={creds is not None}, valid={creds.valid if creds else 'N/A'}")
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("⏳ [DEBUG] Token 已過期，嘗試更新...")
            try:
                creds.refresh(Request())
                print("✅ Token 已更新")
            except Exception as e:
                print(f"❌ Token 更新失敗：{e}")
                creds = None
        
        if not creds:
            print(f"📍 [DEBUG] 方法 3: 檢查 OAuth credentials.json ({credentials_file})")
            if not os.path.exists(credentials_file):
                print(f"❌ 找不到 OAuth 憑證檔案：{credentials_file}")
                print("⚠️ 所有授權方式都失敗了！")
                print("解決方案：")
                print("  - Streamlit Cloud: 設定 Secrets 中的 oauth_token")
                print("  - 本地: 執行 python google_drive_utils.py 進行授權")
                print("="*60 + "\n")
                return None
            
            try:
                flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
                # 使用 run_local_server，如果失敗會自動降級到 run_console
                try:
                    creds = flow.run_local_server(port=0, open_browser=True)
                    print("✅ 授權成功！")
                except Exception as local_error:
                    print(f"⚠️ 本地伺服器授權失敗，改用手動模式：{local_error}")
                    print("\n請複製以下網址到瀏覽器開啟：")
                    creds = flow.run_console()
                    print("✅ 授權成功！")
            except Exception as e:
                print(f"❌ OAuth 授權失敗：{e}")
                return None
        
        # 儲存 token 以供下次使用
        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)
            print(f"✅ Token 已儲存到 {token_file}")
    
    print("🎯 [DEBUG] 準備建立 Drive service...")
    try:
        service = build('drive', 'v3', credentials=creds)
        print("✅ Google Drive service 建立成功！")
        print("="*60 + "\n")
        return service
    except Exception as e:
        print(f"❌ 建立 Drive service 失敗：{e}")
        print("="*60 + "\n")
        return None

def upload_to_drive(service, file_path: Path, folder_id: Optional[str] = None, 
                    mime_type: str = 'application/json') -> Optional[str]:
    """
    上傳檔案到 Google Drive
    
    Args:
        service: Google Drive API service 物件
        file_path: 要上傳的檔案路徑
        folder_id: 目標資料夾 ID（可選）
        mime_type: 檔案 MIME type
        
    Returns:
        上傳後的檔案 ID，失敗則返回 None
    """
    if not service:
        return None
    
    try:
        file_metadata = {'name': file_path.name}
        if folder_id:
            file_metadata['parents'] = [folder_id]
        
        media = MediaFileUpload(str(file_path), mimetype=mime_type, resumable=True)
        
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, webViewLink'
        ).execute()
        
        file_id = file.get('id')
        web_link = file.get('webViewLink')
        
        print(f"✅ 檔案已上傳：{file_path.name}")
        print(f"   Drive ID: {file_id}")
        print(f"   連結: {web_link}")
        
        return file_id
        
    except HttpError as error:
        print(f"❌ 上傳失敗：{error}")
        return None
    except Exception as e:
        print(f"❌ 上傳過程發生錯誤：{e}")
        return None

def test_drive_connection(credentials_file: str = 'credentials.json'):
    """
    測試 Google Drive 連線
    """
    print("=== 測試 Google Drive 連線 ===\n")
    
    service = get_drive_service(credentials_file)
    
    if service:
        try:
            # 列出 Drive 中的前 5 個檔案
            results = service.files().list(pageSize=5, fields="files(id, name)").execute()
            items = results.get('files', [])
            
            if not items:
                print('沒有找到任何檔案')
            else:
                print('你的 Drive 中的檔案（前 5 個）：')
                for item in items:
                    print(f"  - {item['name']} ({item['id']})")
            
            print("\n✅ Google Drive 連線測試成功！")
            return True
            
        except Exception as e:
            print(f"❌ 列出檔案時發生錯誤：{e}")
            return False
    else:
        print("❌ 無法連接到 Google Drive")
        return False

if __name__ == "__main__":
    # 測試連線
    test_drive_connection()
