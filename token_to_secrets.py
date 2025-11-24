#!/usr/bin/env python3
"""
將 token.pickle 轉換為 Streamlit secrets.toml 格式
用於 Streamlit Cloud 部署
"""
import pickle
import sys
from pathlib import Path

def extract_token_info(token_file: str = 'token.pickle'):
    """
    從 token.pickle 提取 OAuth token 資訊
    
    Args:
        token_file: token.pickle 檔案路徑
        
    Returns:
        token 資訊的字典，失敗則返回 None
    """
    try:
        if not Path(token_file).exists():
            print(f"❌ 找不到檔案：{token_file}")
            print("\n請先執行以下步驟：")
            print("1. 確認 credentials.json 存在")
            print("2. 執行：python google_drive_utils.py")
            print("3. 完成瀏覽器授權")
            print("4. 會自動生成 token.pickle")
            return None
        
        # 讀取 pickle 檔案
        with open(token_file, 'rb') as f:
            creds = pickle.load(f)
        
        # 提取必要資訊
        token_info = {
            'token': creds.token,
            'refresh_token': creds.refresh_token,
            'token_uri': creds.token_uri,
            'client_id': creds.client_id,
            'client_secret': creds.client_secret,
            'scopes': list(creds.scopes) if creds.scopes else []
        }
        
        return token_info
        
    except Exception as e:
        print(f"❌ 讀取 token 失敗：{e}")
        return None

def generate_secrets_toml(
    token_info: dict,
    output_file: str = '.streamlit/secrets.toml',
    drive_folder_id: str = '16HRRkutsZcscFkk4Q7XgJPEjbz3nurod'
):
    """
    生成 Streamlit secrets.toml 檔案
    
    Args:
        token_info: token 資訊字典
        output_file: 輸出檔案路徑
        drive_folder_id: Google Drive 資料夾 ID
    """
    try:
        # 建立 .streamlit 目錄
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        # 格式化 scopes
        scopes_str = '\n'.join([f'  "{scope}",' for scope in token_info['scopes']])
        
        # 生成 secrets.toml 內容
        secrets_content = f"""# Streamlit Secrets for Google Drive OAuth Integration
# Auto-generated from token.pickle
# ⚠️ 不要上傳此檔案到 Git！

# Google Drive 資料夾 ID
DRIVE_FOLDER_ID = "{drive_folder_id}"

# OAuth 2.0 Token（從 token.pickle 提取）
[oauth_token]
token = "{token_info['token']}"
refresh_token = "{token_info['refresh_token']}"
token_uri = "{token_info['token_uri']}"
client_id = "{token_info['client_id']}"
client_secret = "{token_info['client_secret']}"
scopes = [
{scopes_str}
]
"""
        
        # 寫入檔案
        with open(output_file, 'w') as f:
            f.write(secrets_content)
        
        print(f"✅ secrets.toml 已生成：{output_file}")
        print("\n📋 Token 資訊：")
        print(f"   Client ID: {token_info['client_id'][:20]}...")
        print(f"   Token URI: {token_info['token_uri']}")
        print(f"   Scopes: {', '.join(token_info['scopes'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ 生成 secrets.toml 失敗：{e}")
        return False

def print_deployment_instructions(secrets_file: str):
    """顯示部署說明"""
    print("\n" + "=" * 60)
    print("🚀 Streamlit Cloud 部署步驟")
    print("=" * 60)
    print("\n1️⃣  本地測試：")
    print("   streamlit run app_emotion_guided.py")
    print("   確認能正常連線到 Google Drive")
    print()
    print("2️⃣  部署到 Streamlit Cloud：")
    print("   a. 推送程式碼到 GitHub")
    print("   b. 在 Streamlit Cloud 建立新應用程式")
    print("   c. 開啟 App Settings > Secrets")
    print(f"   d. 複製 {secrets_file} 的完整內容")
    print("   e. 貼到 Secrets 欄位")
    print("   f. 點擊 Save")
    print()
    print("3️⃣  驗證：")
    print("   - 應用程式會自動重啟")
    print("   - 檢查 logs 確認 '✅ 使用 Streamlit Secrets 中的 OAuth token'")
    print("   - 完成對話並評分，確認檔案上傳到 Drive")
    print()
    print("⚠️  重要提醒：")
    print("   - secrets.toml 不要上傳到 GitHub（已加入 .gitignore）")
    print("   - Token 有效期約 7 天，會自動更新")
    print("   - 如果 token 失效，重新執行此腳本")
    print()

def main():
    """主程式"""
    print("=" * 60)
    print("Token to Streamlit Secrets 轉換工具")
    print("=" * 60)
    print()
    
    # 取得參數
    token_file = 'token.pickle'
    drive_folder_id = '16HRRkutsZcscFkk4Q7XgJPEjbz3nurod'
    
    if len(sys.argv) > 1:
        token_file = sys.argv[1]
    
    if len(sys.argv) > 2:
        drive_folder_id = sys.argv[2]
    
    print(f"輸入檔案: {token_file}")
    print(f"Drive 資料夾 ID: {drive_folder_id}")
    print()
    
    # 提取 token 資訊
    print("📖 讀取 token.pickle...")
    token_info = extract_token_info(token_file)
    
    if not token_info:
        print("\n❌ 轉換失敗")
        sys.exit(1)
    
    # 生成 secrets.toml
    print("\n📝 生成 secrets.toml...")
    secrets_file = '.streamlit/secrets.toml'
    success = generate_secrets_toml(token_info, secrets_file, drive_folder_id)
    
    if success:
        print_deployment_instructions(secrets_file)
        print("✨ 完成！")
    else:
        print("\n❌ 轉換失敗")
        sys.exit(1)

if __name__ == "__main__":
    main()
