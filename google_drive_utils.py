"""
Google Drive 工具模組
支援 OAuth 2.0 使用者授權（個人 Google Drive）
"""
import os
import pickle
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

# Google Drive API 權限範圍
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def get_drive_service(credentials_file: str = 'credentials.json', token_file: str = 'token.pickle'):
    """
    初始化 Google Drive API service（使用 OAuth 2.0）
    
    Args:
        credentials_file: OAuth 2.0 憑證檔案路徑（從 GCP Console 下載）
        token_file: Token 快取檔案路徑（自動生成）
        
    Returns:
        Google Drive API service 物件，如果失敗則返回 None
    """
    if not GOOGLE_DRIVE_AVAILABLE:
        print("⚠️ Google Drive API libraries 未安裝")
        return None
    
    creds = None
    
    # 檢查是否有已儲存的 token
    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)
    
    # 如果沒有有效的憑證，需要重新登入
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                print("✅ Token 已更新")
            except Exception as e:
                print(f"⚠️ Token 更新失敗：{e}")
                creds = None
        
        if not creds:
            if not os.path.exists(credentials_file):
                print(f"❌ 找不到 OAuth 憑證檔案：{credentials_file}")
                print("請從 Google Cloud Console 下載 OAuth 2.0 憑證並命名為 credentials.json")
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
    
    try:
        service = build('drive', 'v3', credentials=creds)
        return service
    except Exception as e:
        print(f"❌ 建立 Drive service 失敗：{e}")
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
