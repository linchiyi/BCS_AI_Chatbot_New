"""
Session Logger & Google Drive Uploader
用於記錄對話 session 到本地 JSON 並上傳到 Google Drive
支援 OAuth 2.0 使用者授權
"""
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, List, Optional

# 使用新的 google_drive_utils 模組
try:
    from google_drive_utils import get_drive_service, upload_to_drive, GOOGLE_DRIVE_AVAILABLE
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False
    get_drive_service = None
    upload_to_drive = None


class SessionLogger:
    """管理對話 session 的本地記錄與 Google Drive 上傳"""
    
    def __init__(self, logs_dir: Path, drive_folder_id: Optional[str] = None):
        """
        初始化 SessionLogger
        
        Args:
            logs_dir: 本地 logs 資料夾路徑
            drive_folder_id: Google Drive 目標資料夾 ID（可選）
        """
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(exist_ok=True)
        self.drive_folder_id = drive_folder_id
        self.drive_service = None
        self.tz = ZoneInfo("Asia/Taipei")
        
        # 嘗試初始化 Google Drive service
        if GOOGLE_DRIVE_AVAILABLE and drive_folder_id:
            self._init_drive_service()

    @staticmethod
    def _strip_visual_tags(content: str) -> str:
        """移除情緒卡片/HTML，保留純文字"""
        import re
        if not content:
            return ""
        text = re.sub(r"<[^>]+>", "", content)
        lines = []
        for line in text.splitlines():
            if any(key in line for key in ("情緒狀態：", "強度：", "💭", "Emotion State:", "Intensity:")):
                continue
            lines.append(line)
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text
    
    def _init_drive_service(self):
        """
        初始化 Google Drive API service
        自動偵測並使用最適合的授權方式：
        1. Streamlit Cloud: 使用 st.secrets 中的 OAuth token
        2. 本地 (有 token.pickle): 使用已授權的 OAuth token
        3. 本地 (僅有 credentials.json): 需要瀏覽器授權一次
        """
        print("\n" + "🔵"*30)
        print("📝 SessionLogger: 初始化 Google Drive service")
        print("🔵"*30)
        
        if not GOOGLE_DRIVE_AVAILABLE or not get_drive_service:
            print("❌ Google Drive 功能不可用")
            print("   - GOOGLE_DRIVE_AVAILABLE:", GOOGLE_DRIVE_AVAILABLE)
            print("   - get_drive_service:", get_drive_service)
            return
        
        try:
            # 直接呼叫 get_drive_service，它會自動偵測環境
            # 並選擇最適合的授權方式（Secrets > token.pickle > credentials.json）
            print("🚀 呼叫 get_drive_service()...")
            self.drive_service = get_drive_service()
            
            if self.drive_service:
                print("✅ ✅ ✅ Google Drive service 初始化成功！")
                print(f"📁 目標資料夾 ID: {self.drive_folder_id}")
            else:
                print("❌ ❌ ❌ Google Drive service 初始化失敗")
                print("   可能原因：")
                print("   - Streamlit Cloud: 需要在 Settings > Secrets 設定 oauth_token")
                print("   - 本地開發: 需要 token.pickle 或 credentials.json")
            
            print("🔵"*30 + "\n")
                
        except Exception as e:
            print(f"❌ ❌ ❌ Google Drive service 初始化失敗：{e}")
            import traceback
            traceback.print_exc()
            self.drive_service = None
            print("🔵"*30 + "\n")
    
    def log_session(
        self,
        messages: List[Dict[str, str]],
        evaluation: Optional[Dict[str, Any]],
        stage: str,
        emotion_mode: str,
        student_level: int,
        shair_feedback: str,
        conversation_seconds: int,
        diagnosis_disclosed: bool,
        case_id: str = "",
        case_name: str = "",
        user_info: Optional[Dict[str, str]] = None,
    ) -> Optional[Path]:
        """
        記錄一個完整的對話 session 到本地 JSON 檔案
        
        Args:
            user_info: 使用者資訊（identity, group, serial）
        
        Returns:
            記錄檔案的路徑，若失敗則回傳 None
        """
        try:
            now = datetime.now(self.tz)
            timestamp = now.strftime("%Y%m%d_%H%M%S")

            # 先淨化訊息，移除情緒卡片/HTML
            cleaned_messages: List[Dict[str, str]] = []
            for msg in messages:
                content = msg.get("content", "")
                cleaned = self._strip_visual_tags(content)
                cleaned_messages.append({**msg, "content": cleaned})
            
            # 根據教案產生檔名前綴
            case_prefix = ""
            if case_id == "npc":
                case_prefix = "鼻咽癌_"
            elif case_id == "abdominal_pain":
                case_prefix = "腹痛_"
            
            # 加入使用者資訊到檔名
            user_suffix = ""
            if user_info:
                identity = user_info.get("identity", "")
                group = user_info.get("group", "")
                serial = user_info.get("serial", "")
                if identity or group or serial:
                    user_suffix = f"_{identity}_{group}_{serial}"
            
            filename = self.logs_dir / f"{case_prefix}session_{timestamp}{user_suffix}.json"
            
            payload = {
                "timestamp": timestamp,
                "datetime": now.isoformat(),
                "case_id": case_id,
                "case_name": case_name,
                "user_info": user_info or {},
                "student_level": student_level,
                "emotion_mode": emotion_mode,
                "stage": stage,
                "diagnosis_disclosed": diagnosis_disclosed,
                "conversation_seconds": conversation_seconds,
                "conversation_minutes": round(conversation_seconds / 60, 2),
                "messages": cleaned_messages,
                "evaluation": evaluation.get("structured") if evaluation else None,
                "evaluation_raw": evaluation.get("raw_text") if evaluation else None,
                "shair_feedback": shair_feedback,
            }
            
            with filename.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Session 記錄已儲存至：{filename}")
            return filename
            
        except Exception as exc:
            print(f"⚠️ 儲存 session 記錄失敗：{exc}")
            return None
    
    def upload_to_drive(self, local_file: Path, remote_filename: Optional[str] = None) -> Optional[str]:
        """
        上傳檔案到 Google Drive
        
        Args:
            local_file: 本地檔案路徑
            remote_filename: 遠端檔案名稱（可選，預設使用本地檔名）
        
        Returns:
            上傳成功的檔案 ID，失敗則回傳 None
        """
        if not self.drive_service:
            return None
        
        if not self.drive_folder_id:
            print("⚠️ 未設定 Drive 資料夾 ID")
            return None
        
        try:
            # 使用 google_drive_utils 的上傳函數
            mime_type = 'application/json' if local_file.suffix == '.json' else 'text/plain'
            file_id = upload_to_drive(
                service=self.drive_service,
                file_path=local_file,
                folder_id=self.drive_folder_id,
                mime_type=mime_type
            )
            return file_id
            
        except Exception as exc:
            print(f"⚠️ 上傳到 Google Drive 失敗：{exc}")
            return None
    
    def log_and_upload(
        self,
        messages: List[Dict[str, str]],
        evaluation: Optional[Dict[str, Any]],
        stage: str,
        emotion_mode: str,
        student_level: int,
        shair_feedback: str,
        conversation_seconds: int,
        diagnosis_disclosed: bool,
        combined_report_bytes: bytes,
        case_id: str = "",
        case_name: str = "",
        user_info: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        記錄 session 並上傳到 Google Drive
        
        Args:
            case_id: 教案識別碼（如 'npc', 'abdominal_pain'）
            case_name: 教案名稱（如 '鼻咽癌 - 病情告知'）
            user_info: 使用者資訊（identity, group, serial）
        
        Returns:
            包含 local_path, drive_file_id, report_drive_id 的字典
        """
        result = {
            "local_path": None,
            "drive_file_id": None,
            "report_drive_id": None,
            "error_message": None,
        }
        
        # 根據教案產生檔名前綴
        case_prefix = ""
        if case_id == "npc":
            case_prefix = "鼻咽癌_"
        elif case_id == "abdominal_pain":
            case_prefix = "腹痛_"
        
        # 加入使用者資訊到檔名
        user_suffix = ""
        if user_info:
            identity = user_info.get("identity", "")
            group = user_info.get("group", "")
            serial = user_info.get("serial", "")
            if identity or group or serial:
                user_suffix = f"_{identity}_{group}_{serial}"
        
        # 1. 儲存本地 JSON log
        local_path = self.log_session(
            messages=messages,
            evaluation=evaluation,
            stage=stage,
            emotion_mode=emotion_mode,
            student_level=student_level,
            shair_feedback=shair_feedback,
            conversation_seconds=conversation_seconds,
            diagnosis_disclosed=diagnosis_disclosed,
            case_id=case_id,
            case_name=case_name,
            user_info=user_info,
        )
        result["local_path"] = str(local_path) if local_path else None
        
        # 2. 上傳 JSON log 到 Drive
        if local_path and self.drive_service:
            drive_id = self.upload_to_drive(local_path)
            result["drive_file_id"] = drive_id
        
        # 3. 儲存並上傳 combined report (txt)
        if combined_report_bytes:
            try:
                timestamp = datetime.now(self.tz).strftime("%Y%m%d_%H%M%S")
                report_filename = self.logs_dir / f"{case_prefix}report_{timestamp}{user_suffix}.txt"
                report_filename.write_bytes(combined_report_bytes)
                
                if self.drive_service:
                    report_drive_id = self.upload_to_drive(report_filename)
                    result["report_drive_id"] = report_drive_id
                    
            except Exception as exc:
                print(f"⚠️ 儲存或上傳 combined report 失敗：{exc}")
                result["error_message"] = str(exc)
        
        return result
