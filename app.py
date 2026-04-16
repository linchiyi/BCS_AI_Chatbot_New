"""
OSCE 醫病對話模擬器 - 多教案整合版
在進入對話前選擇教案，每個教案有獨立的 context engine 確保不會互相影響
支援語音對話：語音輸入（Whisper）+ 語音回覆（TTS）
"""

import csv
import io
import json
import os
import sys
import time
import tempfile
import hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI

try:
    import pandas as pd
except ImportError:
    pd = None

# =========================================================
# 頁面設定
# =========================================================
st.set_page_config(
    page_title="OSCE 醫病對話模擬器",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded",
)

# =========================================================
# 環境與 OpenAI 初始化
# =========================================================
load_dotenv()
# 伺服端預設 API Key（Email 模式會用這支；API Key 模式由使用者輸入）
SERVER_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("PATIENT_MODEL", "gpt-4.1")
EMBEDDING_MODEL = os.getenv("PATIENT_EMBEDDING_MODEL", "text-embedding-3-large")
EVALUATION_MODEL = os.getenv("PATIENT_EVALUATION_MODEL", "gpt-4.1")
ADMIN_ACCESS_CODE = os.getenv("CHATBOT_ADMIN_CODE", "")

PROJECT_ROOT = Path(__file__).resolve().parent
TZ = ZoneInfo("Asia/Taipei")

# =========================================================
# 教案選項
# =========================================================
CASE_OPTIONS = {
    "npc": {
        "name": "鼻咽癌 - 病情告知",
        "icon": "🩺",
        "description": "55 歲男性病人吳忠明，回診確認鼻咽癌病理報告。練習告知壞消息與情緒處理。",
        "role": "病人",
        "patient_name": "吳忠明",
        "scenario": "門診",
        "avatar_patient": "🤒",
    },
    "abdominal_pain": {
        "name": "腹痛 - 家屬溝通",
        "icon": "🚑",
        "description": "75 歲男性病人陳志華，腹膜透析患者因腹痛送急診。與家屬（長女）溝通病情與治療選項。",
        "role": "家屬（長女）",
        "patient_name": "陳志華",
        "scenario": "急診",
        "avatar_patient": "👩",
    },
}

# =========================================================
# 暫時停用的教案（把 case_id 加進這個 set 就會隱藏該教案按鈕）
# 要重新啟用時只要把該 id 從 set 中移除即可
# 重新開放腹痛教案 → 把 "abdominal_pain" 從 set 裡移除，改成 DISABLED_CASES = set() , 改成只留腹痛 → 改成 DISABLED_CASES = {"npc"}
# =========================================================
DISABLED_CASES = set()   # ← 目前關閉腹痛教案；全開請改成 set()

# =========================================================
# Session State 初始化
# =========================================================
if "selected_case" not in st.session_state:
    st.session_state.selected_case = None
if "case_confirmed" not in st.session_state:
    st.session_state.case_confirmed = False
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "api_key"  # api_key | email
if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False
if "auth_user_email" not in st.session_state:
    st.session_state.auth_user_email = ""
if "active_api_key" not in st.session_state:
    st.session_state.active_api_key = ""

# 使用者身分相關的 session state
if "user_identity" not in st.session_state:
    st.session_state.user_identity = "醫學生"
if "user_group" not in st.session_state:
    st.session_state.user_group = "第1組"
if "user_serial" not in st.session_state:
    st.session_state.user_serial = "1"

# 語音模式相關的 session state
if "voice_mode" not in st.session_state:
    st.session_state.voice_mode = False  # False=文字模式, True=即時語音模式
if "voice_input_mode" not in st.session_state:
    st.session_state.voice_input_mode = False  # 語音輸入模式（先轉文字再送出）
if "voice_messages" not in st.session_state:
    st.session_state.voice_messages = []
if "voice_duration" not in st.session_state:
    st.session_state.voice_duration = 0
if "voice_conversation_ended" not in st.session_state:
    st.session_state.voice_conversation_ended = False
if "voice_selected" not in st.session_state:
    st.session_state.voice_selected = "shimmer"
if "voice_input_text" not in st.session_state:
    st.session_state.voice_input_text = ""  # 語音輸入模式的暫存文字
if "pending_tts_audio" not in st.session_state:
    st.session_state.pending_tts_audio = None  # 待播放的 TTS 音頻


def reset_to_case_selection():
    """返回教案選擇頁面"""
    st.session_state.selected_case = None
    st.session_state.case_confirmed = False
    st.session_state.is_authenticated = False
    st.session_state.auth_user_email = ""
    st.session_state.active_api_key = ""
    st.session_state.auth_mode = "api_key"
    # 清除其他對話相關的 session state
    keys_to_clear = [
        "messages", "emotion_mode", "stage", "student_level",
        "last_evaluation", "last_evaluation_error", "pending_evaluation",
        "diagnosis_disclosed", "conversation_started_at", "timer_frozen_at",
        "timeout_triggered", "logged_this_session", "admin_mode",
        "context_engine", "case_config",
        # 語音模式相關
        "voice_mode", "voice_input_mode", "voice_messages", "voice_duration",
        "voice_conversation_ended", "voice_selected", "voice_input_text",
        "pending_tts_audio",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def reset_voice_mode():
    """重置語音模式對話"""
    st.session_state.voice_messages = []
    st.session_state.voice_duration = 0
    st.session_state.voice_conversation_ended = False
    st.session_state.last_evaluation = None
    st.session_state.last_evaluation_error = None
    st.session_state.steps_feedback = None
    st.session_state.spikes_feedback = None
    st.session_state.shair_feedback = None


def _hash_password(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verify_user(email: str, password: str) -> bool:
    users = {}
    try:
        users = st.secrets.get("auth_users", {}) or {}
    except Exception:
        users = {}
    stored = users.get(email)
    if not stored:
        return False
    pw = password or ""
    if stored.startswith("sha256:"):
        return _hash_password(pw) == stored.replace("sha256:", "", 1)
    if stored.startswith("plain:"):
        return pw == stored.replace("plain:", "", 1)
    return pw == stored


# =========================================================
# 教案選擇頁面
# =========================================================
if not st.session_state.case_confirmed:
    st.title("🏥 OSCE 醫病對話模擬器")
    st.markdown("---")
    st.subheader("🔐 登入方式")
    st.session_state.auth_mode = st.radio(
        "選擇登入方式",
        options=["api_key", "email"],
        format_func=lambda x: "API Key" if x == "api_key" else "Email",
        horizontal=True,
    )

    if st.session_state.auth_mode == "api_key":
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_api_key if st.session_state.is_authenticated else "",
            type="password",
            help="僅本機使用，不會寫入檔案。",
        ).strip()
        if st.button("使用此 API Key", type="primary"):
            if st.session_state.openai_api_key:
                st.session_state.active_api_key = st.session_state.openai_api_key
                st.session_state.is_authenticated = True
                st.session_state.auth_user_email = ""
                st.success("已啟用 API Key 模式")
            else:
                st.error("請輸入有效的 API Key")
    else:
        if not SERVER_API_KEY:
            st.error("伺服端未設定 OPENAI_API_KEY，無法使用 Email 登入。請改用 API Key 模式。")
        email = st.text_input("Email", value=st.session_state.auth_user_email)
        password = st.text_input("密碼", type="password")
        if st.button("登入", type="primary"):
            if not email or not password:
                st.warning("請輸入 Email 與密碼")
            elif verify_user(email, password):
                st.session_state.is_authenticated = True
                st.session_state.auth_user_email = email
                st.session_state.active_api_key = SERVER_API_KEY
                st.success(f"已以 {email} 登入，小組金鑰已啟用")
            else:
                st.error("帳號或密碼錯誤")

    if st.session_state.is_authenticated:
        st.info(
            f"登入方式：{'API Key' if st.session_state.auth_mode == 'api_key' else 'Email'}"
            + (f"｜使用者：{st.session_state.auth_user_email}" if st.session_state.auth_user_email else "")
        )
        if st.button("登出", type="secondary"):
            st.session_state.is_authenticated = False
            st.session_state.auth_user_email = ""
            st.session_state.active_api_key = ""
            st.session_state.auth_mode = "api_key"
            st.rerun()

    # 使用者身分選單
    st.subheader("👤 使用者資訊")
    user_cols = st.columns(3)
    with user_cols[0]:
        st.session_state.user_identity = st.selectbox(
            "使用者身分",
            options=["醫學生", "臨床教師", "測試者", "其他"],
            index=["醫學生", "臨床教師", "測試者", "其他"].index(st.session_state.user_identity),
            help="請選擇您的身分類別"
        )
    with user_cols[1]:
        group_options = [f"第{i}組" for i in range(1, 19)]
        st.session_state.user_group = st.selectbox(
            "組別",
            options=group_options,
            index=group_options.index(st.session_state.user_group) if st.session_state.user_group in group_options else 0,
            help="請選擇您的組別（第1組~第18組）"
        )
    with user_cols[2]:
        serial_options = [str(i) for i in range(1, 11)]
        st.session_state.user_serial = st.selectbox(
            "序號",
            options=serial_options,
            index=serial_options.index(st.session_state.user_serial) if st.session_state.user_serial in serial_options else 0,
            help="請選擇您的序號（1-10）"
        )

    st.markdown("---")
    st.subheader("請選擇練習教案")
    st.markdown("每個教案有獨立的對話情境和評分標準。選擇後將進入對應的模擬對話。")
    st.markdown("")
    
    # 教案選擇卡片（排除停用教案）
    active_cases = [(cid, ci) for cid, ci in CASE_OPTIONS.items() if cid not in DISABLED_CASES]
    cols = st.columns(min(2, len(active_cases)))
    
    for idx, (case_id, case_info) in enumerate(active_cases):
        with cols[idx % len(cols)]:
            with st.container(border=True):
                st.markdown(f"### {case_info['icon']} {case_info['name']}")
                st.markdown(f"**角色：** {case_info['role']}")
                st.markdown(f"**病人：** {case_info['patient_name']}")
                st.markdown(f"**場景：** {case_info['scenario']}")
                st.markdown(f"")
                st.caption(case_info['description'])
                st.markdown("")
                if st.button(
                    f"選擇此教案",
                    key=f"select_{case_id}",
                    type="primary",
                    use_container_width=True,
                    disabled=not st.session_state.is_authenticated or not st.session_state.active_api_key,
                ):
                    st.session_state.selected_case = case_id
                    st.session_state.case_confirmed = True
                    st.rerun()
    
    # st.markdown("---")
    # st.caption("💡 提示：每個教案的對話紀錄和評分是獨立的，不會互相影響。")
    st.stop()

# =========================================================
# 以下是選擇教案後的對話邏輯
# =========================================================

# 取得實際使用的 API Key（依登入模式）
API_KEY = (st.session_state.get("active_api_key") or "").strip()
if not API_KEY or not st.session_state.is_authenticated:
    st.error("❌ 尚未登入或未設定 API Key，請返回上一頁完成登入。")
    st.stop()

try:
    client = OpenAI(api_key=API_KEY)
except Exception as exc:
    st.error(f"初始化 OpenAI client 失敗：{exc}")
    st.stop()

selected_case = st.session_state.selected_case
case_info = CASE_OPTIONS.get(selected_case, {})

# =========================================================
# 根據教案載入對應配置
# =========================================================
if selected_case == "npc":
    from cases.case_npc import (
        PATIENT_PERSONA,
        EMOTION_MODES,
        STAGES,
        STAGE_GUIDANCE,
        STAGE_SAFEGUARDS,
        DIAGNOSIS_KEY_TERMS,
        EVALUATION_SYSTEM_PROMPT,
        compose_system_prompt as case_compose_system_prompt,
    )
    from patient_context_engine import PatientContextEngine
    from session_logger import SessionLogger
    
    # 載入 context engine（只載入鼻咽癌語料）
    DEFAULT_SCRIPT_FILES = [
        PROJECT_ROOT.parent / "llm_medical_simulator" / "醫三-五年級的對話腳本_1.txt",
        PROJECT_ROOT.parent / "llm_medical_simulator" / "醫三-五年級的對話腳本_2.txt",
    ]
    DEFAULT_TRANSCRIPTS_DIR = PROJECT_ROOT.parent / "llm_medical_simulator" / "逐字稿_cleaned"
    
    @st.cache_resource(show_spinner=False)
    def load_npc_context_engine():
        existing_scripts = [p for p in DEFAULT_SCRIPT_FILES if p.exists()]
        return PatientContextEngine(
            script_paths=existing_scripts,
            transcripts_dir=DEFAULT_TRANSCRIPTS_DIR if DEFAULT_TRANSCRIPTS_DIR.exists() else None,
            transcript_limit=4,
            transcript_chars=1600,
        )
    
    context_engine = load_npc_context_engine()
    ROLE_LABEL = "病人"
    AVATAR_PATIENT = "🤒"
    HAS_DIAGNOSIS_DISCLOSURE = True
    
elif selected_case == "abdominal_pain":
    from cases.case_abdominal_pain import (
        PATIENT_PERSONA,
        EMOTION_MODES,
        STAGES,
        STAGE_GUIDANCE,
        STAGE_SAFEGUARDS,
        EVALUATION_SYSTEM_PROMPT,
        compose_system_prompt as case_compose_system_prompt,
        TRANSCRIPTS_DIR,
        CONTEXT_EMBEDDINGS_PATH,
        LAB_DATA,
        CT_IMAGES,
    )
    from session_logger import SessionLogger
    
    # 載入腹痛教案的 context engine（使用本地複製的模組）
    try:
        from abdominal_pain_simulator.context_engine import AbdominalPainContextEngine
        
        @st.cache_resource(show_spinner=False)
        def load_abdominal_pain_context_engine():
            return AbdominalPainContextEngine(
                transcripts_dir=TRANSCRIPTS_DIR,
                transcript_limit=4,
                transcript_chars=1600,
            )
        
        context_engine = load_abdominal_pain_context_engine()
    except ImportError as e:
        context_engine = None
        st.warning(f"⚠️ 無法載入腹痛教案的語料引擎：{e}")
    
    ROLE_LABEL = "家屬"
    AVATAR_PATIENT = "👩"
    HAS_DIAGNOSIS_DISCLOSURE = False
    DIAGNOSIS_KEY_TERMS = []
else:
    st.error("❌ 未知的教案選項")
    reset_to_case_selection()
    st.stop()

# =========================================================
# Session Logger 初始化
# =========================================================
# 預設 Drive 資料夾（若 Cloud Secrets/環境變數未提供，至少不會完全停用上傳）
DEFAULT_DRIVE_FOLDER_ID = "16HRRkutsZcscFkk4Q7XgJPEjbz3nurod"

try:
    DRIVE_FOLDER_ID = st.secrets.get(
        "DRIVE_FOLDER_ID",
        os.getenv("GOOGLE_DRIVE_FOLDER_ID", DEFAULT_DRIVE_FOLDER_ID),
    )
except:
    DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", DEFAULT_DRIVE_FOLDER_ID)

if not DRIVE_FOLDER_ID:
    st.warning("⚠️ 未設定 Google Drive 資料夾 ID（DRIVE_FOLDER_ID / GOOGLE_DRIVE_FOLDER_ID），將無法自動上傳。")

LOGS_DIR = PROJECT_ROOT / "logs"

@st.cache_resource(show_spinner=False)
def get_session_logger():
    return SessionLogger(logs_dir=LOGS_DIR, drive_folder_id=DRIVE_FOLDER_ID or None)

session_logger = get_session_logger()

# =========================================================
# 對話相關 Session State
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "emotion_mode" not in st.session_state:
    st.session_state.emotion_mode = list(EMOTION_MODES.keys())[0]
if "stage" not in st.session_state:
    st.session_state.stage = STAGES[0]
if "student_level" not in st.session_state:
    st.session_state.student_level = 3
if "last_evaluation" not in st.session_state:
    st.session_state.last_evaluation = None
if "last_evaluation_error" not in st.session_state:
    st.session_state.last_evaluation_error = None
if "pending_evaluation" not in st.session_state:
    st.session_state.pending_evaluation = False
if "diagnosis_disclosed" not in st.session_state:
    st.session_state.diagnosis_disclosed = False
if "conversation_started_at" not in st.session_state:
    st.session_state.conversation_started_at = None
if "timer_frozen_at" not in st.session_state:
    st.session_state.timer_frozen_at = None
if "timer_limit_minutes" not in st.session_state:
    st.session_state.timer_limit_minutes = 0
if "auto_download_on_timeout" not in st.session_state:
    st.session_state.auto_download_on_timeout = False
if "timeout_triggered" not in st.session_state:
    st.session_state.timeout_triggered = False
if "admin_mode" not in st.session_state:
    st.session_state.admin_mode = False
if "logged_this_session" not in st.session_state:
    st.session_state.logged_this_session = False
if "steps_feedback" not in st.session_state:
    st.session_state.steps_feedback = None
if "spikes_feedback" not in st.session_state:
    st.session_state.spikes_feedback = None
if "shair_feedback" not in st.session_state:
    st.session_state.shair_feedback = None

# =========================================================
# 工具函式
# =========================================================
def get_elapsed_seconds(start_timestamp):
    if not start_timestamp:
        return 0
    end_ts = st.session_state.get("timer_frozen_at") or time.time()
    return max(0, int(end_ts - start_timestamp))


def render_live_timer(start_timestamp: float | None, limit_minutes: int, already_triggered: bool) -> None:
    # 前端僅負責顯示秒數；是否凍結由後端控制 elapsed_seconds
    # 若已凍結，則改用凍結時刻作為結束時間
    if start_timestamp and st.session_state.get("timer_frozen_at"):
        start_ms = int(start_timestamp * 1000)
        frozen_ms = int(st.session_state.timer_frozen_at * 1000)
        # 直接把總秒數固定為凍結時刻的 elapsed，並在前端不再持續累加
        fixed_elapsed_ms = max(0, frozen_ms - start_ms)
    else:
        start_ms = int(start_timestamp * 1000) if start_timestamp else 0
        fixed_elapsed_ms = None
    limit_ms = int(limit_minutes * 60 * 1000) if limit_minutes else 0
    triggered_literal = "true" if already_triggered else "false"
    components.html(
        f"""
        <div class="timer-box">
            <div class="timer-label">對話經過時間</div>
            <div id="timer-display" class="timer-value">00:00</div>
            <div id="timer-limit" class="timer-subtext"></div>
        </div>
        <style>
            .timer-box {{
                padding: 0.5rem 0.75rem;
                border: 1px solid #dddddd;
                border-radius: 0.5rem;
                background-color: #f8f9fa;
            }}
            .timer-label {{
                font-size: 0.85rem;
                color: #555555;
                margin-bottom: 0.15rem;
            }}
            .timer-value {{
                font-size: 1.6rem;
                font-weight: 600;
                color: #1f77b4;
            }}
            .timer-subtext {{
                font-size: 0.8rem;
                color: #6c757d;
                margin-top: 0.2rem;
            }}
            .timer-alert {{
                color: #c82333 !important;
            }}
        </style>
        <script>
            (function() {{
                const displayEl = document.getElementById("timer-display");
                const limitEl = document.getElementById("timer-limit");
                const startMs = {start_ms};
                const limitMs = {limit_ms};
                let timerId = null;
                let hasSignaled = {triggered_literal};

                function updateLimitText(initial) {{
                    if (!limitEl) {{
                        return;
                    }}
                }}

                function formatDuration(ms) {{
                    const totalSeconds = Math.max(0, Math.floor(ms / 1000));
                    const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
                    const seconds = String(totalSeconds % 60).padStart(2, "0");
                    return minutes + ":" + seconds;
                }}

                function updateTimer() {{
                    if (!displayEl) {{
                        return;
                    }}
                    if (startMs <= 0) {{
                        displayEl.textContent = "尚未開始";
                        displayEl.classList.remove("timer-alert");
                        updateLimitText(true);
                        return;
                    }}

                    let elapsed;
                    if ({fixed_elapsed_ms if fixed_elapsed_ms is not None else 'null'} !== null) {{
                        // 已凍結：使用固定 elapsed，不再隨時間增加
                        elapsed = {fixed_elapsed_ms if fixed_elapsed_ms is not None else 0};
                    }} else {{
                        elapsed = Date.now() - startMs;
                    }}
                    displayEl.textContent = formatDuration(elapsed);

                    if (limitMs > 0 && limitEl) {{
                        const remaining = limitMs - elapsed;
                        if (remaining <= 0) {{
                            displayEl.classList.add("timer-alert");
                            limitEl.textContent = "時間已到";
                            clearInterval(timerId);
                            if (!hasSignaled) {{
                                hasSignaled = true;
                                window.parent.postMessage({{
                                    isStreamlitMessage: true,
                                    type: "streamlit:rerun"
                                }}, "*");
                            }}
                        }} else {{
                            const minutesLeft = Math.max(0, Math.floor(remaining / 60000));
                            limitEl.textContent = "剩餘約 " + minutesLeft + " 分";
                            displayEl.classList.remove("timer-alert");
                        }}
                    }} else if (limitEl) {{
                        limitEl.textContent = "不限時";
                    }}
                }}

                updateLimitText(true);
                updateTimer();
                if ({fixed_elapsed_ms if fixed_elapsed_ms is not None else 'null'} === null) {{
                    timerId = setInterval(updateTimer, 1000);
                }}
            }})();
        </script>
        """,
        height=120,
    )


def infer_stage_from_text(text: str, current_stage: str) -> str:
    """根據對話內容推斷階段"""
    t = (text or "").strip()
    if not t:
        return current_stage
    
    if selected_case == "npc":
        # 鼻咽癌教案的階段推斷
        stage3_keywords = ["治療", "追蹤", "下一步", "檢查", "安排"]
        stage2_keywords = ["癌", "報告", "結果", "診斷", "惡性"]
        
        if any(k in t for k in stage3_keywords):
            return STAGES[2] if len(STAGES) > 2 else current_stage
        if any(k in t for k in stage2_keywords):
            return STAGES[1] if len(STAGES) > 1 else current_stage
    
    elif selected_case == "abdominal_pain":
        # 腹痛教案的階段推斷
        stage3_keywords = ["洗手", "無菌", "衛教", "腹膜透析", "換液", "照護", "回家", "注意"]
        stage2_keywords = ["手術", "麻醉", "風險", "併發症", "不開", "不手術", "轉院", "同意", "簽"]
        
        if any(k in t for k in stage3_keywords):
            return STAGES[2] if len(STAGES) > 2 else current_stage
        if any(k in t for k in stage2_keywords):
            return STAGES[1] if len(STAGES) > 1 else current_stage
    
    return current_stage


def update_stage(user_text: str):
    current = st.session_state.stage
    inferred = infer_stage_from_text(user_text, current)
    current_idx = STAGES.index(current)
    inferred_idx = STAGES.index(inferred)
    if inferred_idx > current_idx:
        st.session_state.stage = inferred


def detect_diagnosis_disclosure(user_text: str) -> bool:
    if not HAS_DIAGNOSIS_DISCLOSURE:
        return False
    text = user_text.strip()
    if not text:
        return False
    for term in DIAGNOSIS_KEY_TERMS:
        if term in text or term.lower() in text.lower():
            return True
    if "癌" in text:
        markers = ["確診", "診斷", "報告", "結果", "顯示", "確認", "是", "證實"]
        if any(m in text for m in markers):
            return True
    return False


def get_emotion_visual_config() -> Dict[str, Dict]:
    """
    情緒視覺化配置系統
    包含：表情符號變化、顏色深淺、強度說明
    """
    return {
        # ========== 鼻咽癌教案情緒（完整對應 case_npc.py EMOTION_MODES）==========
        "極度震驚否認型": {
            "base_color": "#B22222",  # 深紅色系 - 震驚
            "emoji_levels": ["😦", "😨", "😱", "😱", "🤯"],
            "intensity_desc": {
                1: "平靜/觀望，日常寒暄",
                2: "感受壞消息暗示，開始追問",
                3: "確認壞消息，開始質疑",
                4: "語無倫次，拒絕接受",
                5: "完全崩潰，大聲否認",
            },
            "color_opacity": [0.2, 0.4, 0.6, 0.8, 1.0],
        },
        "恐懼擔憂型": {
            "base_color": "#FF8C00",  # 橙色系 - 恐懼
            "emoji_levels": ["😟", "😟", "😰", "😰", "😱"],
            "intensity_desc": {
                1: "輕微緊張，等待報告",
                2: "感受暗示，開始追問",
                3: "確認壞消息，焦慮明顯",
                4: "反覆詢問存活率、擔心家人",
                5: "恐慌發作，難以冷靜",
            },
            "color_opacity": [0.2, 0.4, 0.6, 0.8, 1.0],
        },
        "冷靜理性型": {
            "base_color": "#4169E1",  # 藍色系 - 冷靜
            "emoji_levels": ["😐", "🤔", "🤔", "😐", "😐"],
            "intensity_desc": {
                1: "日常問答，配合對話",
                2: "開始詢問治療細節",
                3: "理性分析選項、費用",
                4: "壓抑情緒，條理清晰",
                5: "過度理性，壓抑感受",
            },
            "color_opacity": [0.2, 0.4, 0.6, 0.8, 1.0],
        },
        "悲傷沮喪型": {
            "base_color": "#6A5ACD",  # 紫色系 - 悲傷
            "emoji_levels": ["😔", "😔", "😢", "😢", "😭"],
            "intensity_desc": {
                1: "輕微失落，語氣低落",
                2: "明顯難過，沉默寡言",
                3: "確認壞消息，悲傷明顯",
                4: "聲音哽咽，無力自責",
                5: "崩潰哭泣，覺得絕望",
            },
            "color_opacity": [0.2, 0.4, 0.6, 0.8, 1.0],
        },
        "憤怒質疑型": {
            "base_color": "#DC143C",  # 紅色系 - 憤怒
            "emoji_levels": ["😐", "😒", "😤", "😠", "😡"],
            "intensity_desc": {
                1: "輕微不滿，開始質疑",
                2: "明顯不耐，懷疑檢查",
                3: "態度強硬，要求解釋",
                4: "語氣尖銳，指責醫療",
                5: "激烈抗議，大聲質問",
            },
            "color_opacity": [0.2, 0.4, 0.6, 0.8, 1.0],
        },
        "接受配合型": {
            "base_color": "#32CD32",  # 綠色系 - 接受
            "emoji_levels": ["😐", "🙂", "😊", "💪", "💪"],
            "intensity_desc": {
                1: "被動接受，聽從安排",
                2: "開始配合，願意聽取",
                3: "主動詢問，積極面對",
                4: "態度正向，準備治療",
                5: "完全接受，全力配合",
            },
            "color_opacity": [0.2, 0.4, 0.6, 0.8, 1.0],
        },
        # ========== 腹痛教案情緒（完整對應 case_abdominal_pain.py）==========
        "焦慮擔心": {
            "base_color": "#FF8C00",
            "emoji_levels": ["😟", "😟", "😰", "😰", "😱"],
            "intensity_desc": {
                1: "平靜配合，日常問答",
                2: "開始擔心，追問細節",
                3: "情緒波動，難以專注",
                4: "語調急促，反覆確認",
                5: "極度恐慌，需要冷靜",
            },
            "color_opacity": [0.2, 0.4, 0.6, 0.8, 1.0],
        },
        "自責崩潰": {
            "base_color": "#8B0000",  # 深紅色系
            "emoji_levels": ["😔", "😢", "😢", "😭", "😭"],
            "intensity_desc": {
                1: "輕微自責，語氣低落",
                2: "明顯自責，反覆道歉",
                3: "深度自責，聲音哽咽",
                4: "情緒激動，崩潰哭泣",
                5: "完全崩潰，無法自拔",
            },
            "color_opacity": [0.2, 0.4, 0.6, 0.8, 1.0],
        },
        "憤怒質疑": {
            "base_color": "#DC143C",
            "emoji_levels": ["😒", "😤", "😤", "😠", "😡"],
            "intensity_desc": {
                1: "輕微不滿，偶爾質疑",
                2: "明顯不耐，態度懷疑",
                3: "強烈質疑，要求解釋",
                4: "語氣尖銳，可能打斷",
                5: "激烈抗議，拒絕接受",
            },
            "color_opacity": [0.2, 0.4, 0.6, 0.8, 1.0],
        },
        "堅持轉院": {
            "base_color": "#FF6347",  # 番茄紅
            "emoji_levels": ["🤔", "😤", "😤", "😠", "😠"],
            "intensity_desc": {
                1: "日常問答，尚未提轉院",
                2: "開始考慮，提出疑慮",
                3: "明確傾向，表達想法",
                4: "強烈堅持，難以說服",
                5: "完全拒絕留院，威脅離開",
            },
            "color_opacity": [0.2, 0.4, 0.6, 0.8, 1.0],
        },
        "冷靜配合": {
            "base_color": "#4169E1",
            "emoji_levels": ["😐", "🙂", "🙂", "😊", "😊"],
            "intensity_desc": {
                1: "日常問答，配合對話",
                2: "理解說明，基本配合",
                3: "主動詢問，願意了解",
                4: "積極配合，信任醫師",
                5: "完全信任，全力配合",
            },
            "color_opacity": [0.2, 0.4, 0.6, 0.8, 1.0],
        },
    }


def create_emotion_card_html(emotion_mode: str, intensity: int) -> str:
    """
    建立情緒視覺化 HTML 卡片
    - 背景色深淺代表強度
    - 表情符號隨強度變化
    - 顯示當前強度的行為說明
    """
    config = get_emotion_visual_config()
    emotion_config = config.get(emotion_mode, {
        "base_color": "#888888",
        "emoji_levels": ["😐", "😐", "😐", "😐", "😐"],
        "intensity_desc": {i: "情緒穩定" for i in range(1, 6)},
        "color_opacity": [0.2, 0.4, 0.6, 0.8, 1.0],
    })
    
    # 確保 intensity 在 1-5 範圍
    intensity = max(1, min(5, int(intensity)))
    
    # 取得對應強度的配置
    emoji = emotion_config["emoji_levels"][intensity - 1]
    desc = emotion_config["intensity_desc"].get(intensity, "")
    opacity = emotion_config["color_opacity"][intensity - 1]
    base_color = emotion_config["base_color"]
    
    # 將 hex 顏色轉換為 rgba
    r = int(base_color[1:3], 16)
    g = int(base_color[3:5], 16)
    b = int(base_color[5:7], 16)
    bg_color = f"rgba({r}, {g}, {b}, {opacity})"
    
    # 根據背景深淺決定文字顏色
    text_color = "#FFFFFF" if opacity >= 0.6 else "#333333"
    
    # 建立強度條（填滿的方塊）
    filled_blocks = "█" * intensity
    empty_blocks = "░" * (5 - intensity)
    intensity_bar = filled_blocks + empty_blocks
    
    html = f"""
<div style="
    background: {bg_color};
    border-radius: 10px;
    padding: 12px 16px;
    margin: 10px 0;
    border-left: 4px solid {base_color};
">
    <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 28px;">{emoji}</span>
        <div>
            <div style="color: {text_color}; font-weight: bold; font-size: 14px;">
                情緒狀態：{emotion_mode}
            </div>
            <div style="color: {text_color}; font-size: 13px; margin-top: 2px;">
                強度：<span style="font-family: monospace; letter-spacing: 2px;">{intensity_bar}</span> ({intensity}/5)
            </div>
        </div>
    </div>
    <div style="
        color: {text_color};
        font-size: 12px;
        margin-top: 8px;
        padding-top: 8px;
        border-top: 1px solid rgba(255,255,255,0.3);
        font-style: italic;
    ">
        💭 {desc}
    </div>
</div>
"""
    return html


def annotate_with_intensity(content: str, emotion_mode: str) -> str:
    """
    為回覆加上情緒視覺化標註
    - 解析 AI 回覆中的 {EMOTION:X} 動態強度標記
    - 移除標記並用視覺化卡片顯示
    """
    import re
    
    # 如果已經有視覺化卡片（HTML），直接返回
    if "情緒狀態：" in content and "<div" in content:
        return content
    
    # 解析動態情緒強度標記 {EMOTION:X}
    dynamic_intensity = None
    emotion_match = re.search(r'\{EMOTION:(\d)\}', content)
    if emotion_match:
        dynamic_intensity = int(emotion_match.group(1))
        # 移除標記
        content = re.sub(r'\s*\{EMOTION:\d\}', '', content)
    
    # 移除 AI 可能自己加的舊格式情緒標籤
    clean_content = re.sub(r'\n*【情緒強度：[^】]+】', '', content)
    clean_content = clean_content.strip()
    
    # 決定最終強度：優先使用動態強度，否則使用教案預設
    if dynamic_intensity is not None:
        intensity = dynamic_intensity
    else:
        intensity = EMOTION_MODES.get(emotion_mode, {}).get("intensity", 3)
    
    emotion_html = create_emotion_card_html(emotion_mode, intensity)
    
    return f"{clean_content}\n\n{emotion_html}"


def _strip_visual_tags(content: str) -> str:
    """移除情緒卡片/HTML 標籤，留下純文字對話"""
    import re
    if not content:
        return ""
    text = content
    # 移除 HTML 標籤
    text = re.sub(r"<[^>]+>", "", text)
    # 移除情緒卡片相關行
    lines = []
    for line in text.splitlines():
        if any(key in line for key in ("情緒狀態：", "強度：", "💭")):
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def compose_system_prompt(latest_user_text: str) -> str:
    """組合系統提示詞"""
    emotion_mode = st.session_state.emotion_mode
    stage = st.session_state.stage
    
    # 建立 context block
    context_block = ""
    if context_engine is not None:
        if selected_case == "npc":
            context_block = context_engine.build_context_block(
                level=st.session_state.student_level,
                stage=stage,
                emotion_mode=emotion_mode,
                transcript_chars=1800,
                query_text=latest_user_text,
                embedding_client=client,
                embedding_model=EMBEDDING_MODEL,
            )
        elif selected_case == "abdominal_pain":
            # 使用簡化版本，不做 embedding 查詢，直接使用 sample transcripts
            # 這樣可以避免 API 呼叫延遲
            # 注意：該方法的參數名是 total_chars，不是 transcript_chars
            context_block = context_engine.sample_transcripts(total_chars=1600)
    
    # 使用教案專屬的提示詞組合函式
    if selected_case == "npc":
        return case_compose_system_prompt(
            stage=stage,
            emotion_mode=emotion_mode,
            student_level=st.session_state.student_level,
            context_block=context_block,
            diagnosis_disclosed=st.session_state.diagnosis_disclosed,
        )
    elif selected_case == "abdominal_pain":
        return case_compose_system_prompt(
            stage=stage,
            emotion_mode=emotion_mode,
            context_block=context_block,
        )
    return ""


def _format_conversation_for_model(messages) -> str:
    lines = []
    for idx, msg in enumerate(messages, start=1):
        role = "醫學生" if msg.get("role") == "user" else ROLE_LABEL
        content = _strip_visual_tags(msg.get("content", "").strip())
        lines.append(f"{idx}. {role}: {content}")
    return "\n".join(lines)


def _call_evaluation_api(prompt_text: str) -> str:
    response = client.responses.create(
        model=EVALUATION_MODEL,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": EVALUATION_SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "input_text", "text": prompt_text}]},
        ],
        temperature=0.0,
    )
    collected = []
    for item in getattr(response, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", "") in {"output_text", "text"}:
                collected.append(getattr(c, "text", ""))
    if not collected and hasattr(response, "output_text"):
        collected.append(response.output_text)
    raw = "\n".join(t for t in collected if t).strip()
    if not raw:
        raise RuntimeError("評分模型未返回任何文字內容。")
    return raw


def _parse_evaluation_output(raw_text: str) -> Dict:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass
    first = raw_text.find("{")
    last = raw_text.rfind("}")
    if first != -1 and last != -1 and last > first:
        try:
            return json.loads(raw_text[first:last+1])
        except json.JSONDecodeError:
            pass
    raise ValueError("無法解析評分結果的 JSON。")


def generate_conversation_evaluation(messages) -> Dict:
    if not messages:
        raise ValueError("沒有對話內容可供評分。")
    
    meta_info = f"情緒模式：{st.session_state.emotion_mode}\n對話階段：{st.session_state.stage}\n"
    if selected_case == "npc":
        meta_info += f"醫學生等級：Level {st.session_state.student_level}\n"
    
    conversation_text = _format_conversation_for_model(messages)
    user_prompt = f"""
以下提供一段醫學生與標準化{ROLE_LABEL}的完整逐字稿。
請依據規範輸出單一 JSON 物件，填寫評分項目與整體回饋。

[對話背景]
{meta_info}
[逐字稿]
{conversation_text}
"""
    raw_output = _call_evaluation_api(user_prompt)
    structured = _parse_evaluation_output(raw_output)
    
    # 計算總分
    items = structured.get("evaluation_items", [])
    if isinstance(items, list):
        total = 0
        for item in items:
            if isinstance(item, dict):
                score = item.get("score")
                if isinstance(score, (int, float)):
                    item["score"] = int(score)
                    total += int(score)
        overall = structured.setdefault("overall_performance", {})
        if isinstance(overall, dict):
            overall["total_score"] = total
    
    return {"raw_text": raw_output, "structured": structured}


def request_evaluation():
    st.session_state.pending_evaluation = True


def build_steps_feedback(stage: str, strengths: List[Dict[str, Any]], gaps: List[Dict[str, Any]], conversation_text: str) -> str:
    """產生 STEPS 模式回饋"""
    def join_items(items):
        names = [item.get("項目") for item in items if item.get("項目")]
        return "、".join(names) if names else "尚未顯著項目"

    strength_text = join_items(strengths)
    gap_text = join_items(gaps)

    steps_prompt = f"""
你是一位具溝通教學經驗的 OSCE 主考官，熟悉 STEPS 健康識能溝通技巧：
S = Speak slowly & clearly（說話速度減慢，語調平穩。將訊息以重點、分段，口語化方式說明。每一段落稍微停頓，避免一次給予過多訊息）
T = Teach-back or Show-me（回示教技巧確認病人/家屬的理解程度）
E = Encourage questions（鼓勵病人/家屬問問題）
P = Plain language（用簡單易懂的話，避免醫學術語）
S = Show example（能以舉例、圖片、模型、畫圖或手冊輔助說明）

請根據下列對話逐字稿與評分資訊，以 STEPS 模型對醫學生提供約 400-500 字的中文回饋。

要求：
- 以醫學生為對象，語氣具體、鼓勵且有建設性。
- 請仔細閱讀對話逐字稿，針對醫學生說過的具體句子給出回饋。
- 依序分成五小段輸出，每一段的開頭請明確以「S (Speak slowly & clearly)：」「T (Teach-back)：」「E (Encourage questions)：」「P (Plain language)：」「S (Show example)：」標示。
- 每一段內容約 2-4 句完整句子。

[情境階段]
目前溝通階段：{stage}

[亮點項目]
{strength_text}

[優先改善項目]
{gap_text}

[對話逐字稿]
{conversation_text}
""".strip()

    try:
        response = client.responses.create(
            model=EVALUATION_MODEL,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": "你是臨床溝通技巧教師，熟悉 STEPS 健康識能溝通技巧與 OSCE 評量。"}]},
                {"role": "user", "content": [{"type": "input_text", "text": steps_prompt}]},
            ],
            temperature=0.4,
        )
        collected = []
        for item in getattr(response, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", "") in {"output_text", "text"}:
                    collected.append(getattr(c, "text", ""))
        if not collected and hasattr(response, "output_text"):
            collected.append(response.output_text)
        text = "\n".join(t for t in collected if t).strip()
        if text:
            return text
    except Exception:
        pass
    return f"S (Speak slowly & clearly)：在說明時建議放慢語速、分段說明。\nT (Teach-back)：可嘗試請家屬複述重點，確認理解。\nE (Encourage questions)：主動邀請家屬發問，如「您還有什麼想了解的嗎？」。\nP (Plain language)：對於 {gap_text} 的解釋可用更口語化的方式。\nS (Show example)：可利用圖示或簡單比喻輔助說明。"


def build_spikes_feedback(stage: str, strengths: List[Dict[str, Any]], gaps: List[Dict[str, Any]], conversation_text: str) -> str:
    """產生 SPIKES 模式回饋"""
    def join_items(items):
        names = [item.get("項目") for item in items if item.get("項目")]
        return "、".join(names) if names else "尚未顯著項目"

    strength_text = join_items(strengths)
    gap_text = join_items(gaps)

    spikes_prompt = f"""
你是一位具溝通教學經驗的 OSCE 主考官，熟悉困難溝通中的 SPIKES 模式：
S = Setting（建立關係：環境準備、確認身分、建立信任）
P = Perception（了解病人認知：詢問病人對病情的理解與預期）
I = Invitation（取得病人同意：確認病人想知道多少資訊）
K = Knowledge（說明病情：清楚、分段、避免專有名詞地傳遞壞消息）
E = Empathy（同理心：回應病人情緒、給予支持與陪伴）
S = Strategy and Summary（總結對話：討論後續計畫、確認理解、提供資源）

請根據下列對話逐字稿與評分資訊，以 SPIKES 模型對醫學生提供約 400-500 字的中文回饋。

要求：
- 以醫學生為對象，語氣具體、鼓勵且有建設性。
- 請仔細閱讀對話逐字稿，針對醫學生說過的具體句子給出回饋。
- 依序分成三大段輸出，每一段的開頭請明確標示：
  「一、建立關係 (Setting)：」
  「二、說明解釋 (Perception → Invitation → Knowledge → Empathy)：」
  「三、總結對話 (Strategy and Summary)：」
- 每一段內容約 3-5 句完整句子。

[情境階段]
目前溝通階段：{stage}

[亮點項目]
{strength_text}

[優先改善項目]
{gap_text}

[對話逐字稿]
{conversation_text}
""".strip()

    try:
        response = client.responses.create(
            model=EVALUATION_MODEL,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": "你是臨床溝通技巧教師，熟悉 SPIKES 模型與 OSCE 評量。"}]},
                {"role": "user", "content": [{"type": "input_text", "text": spikes_prompt}]},
            ],
            temperature=0.4,
        )
        collected = []
        for item in getattr(response, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", "") in {"output_text", "text"}:
                    collected.append(getattr(c, "text", ""))
        if not collected and hasattr(response, "output_text"):
            collected.append(response.output_text)
        text = "\n".join(t for t in collected if t).strip()
        if text:
            return text
    except Exception:
        pass
    return f"一、建立關係 (Setting)：目前對話處於「{stage}」階段。\n\n二、說明解釋 (Perception → Invitation → Knowledge → Empathy)：在 {strength_text} 方面表現良好。針對 {gap_text}，建議先了解病人對病情的認知程度。\n\n三、總結對話 (Strategy and Summary)：建議簡要回顧今天討論的重點，確認病人理解程度。"


def build_shair_feedback(stage: str, strengths: List[Dict[str, Any]], gaps: List[Dict[str, Any]], conversation_text: str) -> str:
    """產生 SHAIR 模式回饋"""
    def join_items(items):
        names = [item.get("項目") for item in items if item.get("項目")]
        return "、".join(names) if names else "尚未顯著項目"

    strength_text = join_items(strengths)
    gap_text = join_items(gaps)

    shair_prompt = f"""
你是一位具溝通教學經驗的 OSCE 主考官，熟悉困難溝通中的 SHAIR 模式：
S = Supportive environment（建立支持性的環境與關係）
H = How to deliver（如何傳遞壞消息：語氣、節奏、停頓、用字）
A = Additional information（補充適量且清楚的醫療資訊）
I = Individualize（依病人家庭、身分、價值觀調整說明方式）
R = Reassure and plan（安撫情緒並共同擬定後續計畫）

請根據下列對話逐字稿與評分資訊，以 SHAIR 模型對醫學生提供約 400-500 字的中文回饋。

要求：
- 以醫學生為對象，語氣具體、鼓勵且有建設性。
- 請仔細閱讀對話逐字稿，針對醫學生說過的具體句子給出回饋。
- 依序分成五小段輸出，每一段的開頭請明確以「S (Supportive environment)：」「H (How to deliver)：」「A (Additional information)：」「I (Individualize)：」「R (Reassure and plan)：」標示。
- 每一段內容約 2-4 句完整句子。

[情境階段]
目前溝通階段：{stage}

[亮點項目]
{strength_text}

[優先改善項目]
{gap_text}

[對話逐字稿]
{conversation_text}
""".strip()

    try:
        response = client.responses.create(
            model=EVALUATION_MODEL,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": "你是臨床溝通技巧教師，熟悉 SHAIR 模型與 OSCE 評量。"}]},
                {"role": "user", "content": [{"type": "input_text", "text": shair_prompt}]},
            ],
            temperature=0.4,
        )
        collected = []
        for item in getattr(response, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", "") in {"output_text", "text"}:
                    collected.append(getattr(c, "text", ""))
        if not collected and hasattr(response, "output_text"):
            collected.append(response.output_text)
        text = "\n".join(t for t in collected if t).strip()
        if text:
            return text
    except Exception:
        pass
    return f"S (Supportive environment)：目前對話處於「{stage}」階段。\nH (How to deliver)：你在說明 {strength_text} 時的用字與語氣大致穩定。\nA (Additional information)：對於 {gap_text} 的解釋還可以更具體。\nI (Individualize)：回應時可多連結病人的家庭角色與實際處境。\nR (Reassure and plan)：在安撫情緒的同時，簡要說明下一步安排。"


def build_combined_report(
    messages: List[Dict[str, str]],
    evaluation: Dict[str, Any],
    stage: str,
    emotion_mode: str,
    strengths: List[Dict[str, Any]],
    gaps: List[Dict[str, Any]],
    steps_feedback: str,
    spikes_feedback: str,
    shair_feedback: str,
    case_name: str = "",
    user_info: Dict[str, str] = None,
) -> bytes:
    """建立完整的評分報告"""
    buffer = io.StringIO()
    buffer.write("=== 對話概覽 ===\n")
    if case_name:
        buffer.write(f"教案：{case_name}\n")
    # 使用者資訊
    if user_info:
        if user_info.get("identity"):
            buffer.write(f"使用者身分：{user_info['identity']}\n")
        if user_info.get("group"):
            buffer.write(f"組別：{user_info['group']}\n")
        if user_info.get("number"):
            buffer.write(f"序號：{user_info['number']}\n")
    buffer.write(f"階段：{stage}\n")
    buffer.write(f"情緒模式：{emotion_mode}\n")
    total_seconds = get_elapsed_seconds(st.session_state.conversation_started_at)
    mins = total_seconds // 60
    secs = total_seconds % 60
    buffer.write(f"對話總時長：{mins} 分 {secs} 秒\n")
    buffer.write("\n")
    buffer.write("=== 對話逐字稿 ===\n")
    buffer.write(format_conversation_for_txt(messages))
    buffer.write("\n\n")

    if evaluation:
        structured = evaluation.get("structured", {})
        overall = structured.get("overall_performance", {}) or {}
        buffer.write("=== 評分摘要 ===\n")
        buffer.write(f"項目評分總分：{overall.get('total_score', 'N/A')}\n")
        rating_5 = overall.get("rating_1_to_5", {}) or {}
        r5_score = rating_5.get("score")
        r5_text = "N/A"
        if r5_score is not None:
            try:
                s = int(r5_score)
                mapping = {1: "差", 2: "待加強", 3: "普通", 4: "良好", 5: "優秀"}
                r5_text = f"{s} {mapping.get(s, '')}".strip()
            except:
                r5_text = str(r5_score)
        buffer.write(f"1-5 級整體表現：{r5_text}\n")
        buffer.write(f"重點回饋：{structured.get('brief_feedback', '')}\n\n")

        def _clean_name(n):
            if "." in n:
                parts = n.split(".", 1)
                if parts[0].strip().isdigit():
                    return parts[1].strip()
            return n

        buffer.write("=== 亮點項目 ===\n")
        if strengths:
            for item in strengths:
                buffer.write(f"- {_clean_name(item.get('項目', ''))}\n")
        else:
            buffer.write("- 尚未顯著亮點\n")

        buffer.write("\n=== 待加強項目 ===\n")
        if gaps:
            for item in gaps:
                buffer.write(f"- {_clean_name(item.get('項目', ''))}\n")
        else:
            buffer.write("- 無明顯低分項目\n")

        # 回饋順序：STEPS → SPIKES → SHAIR
        buffer.write("\n=== STEPS 回饋 ===\n")
        buffer.write(steps_feedback)
        buffer.write("\n")

        buffer.write("\n=== SPIKES 回饋 ===\n")
        buffer.write(spikes_feedback)
        buffer.write("\n")

        buffer.write("\n=== SHAIR 回饋 ===\n")
        buffer.write(shair_feedback)
        buffer.write("\n")

    return buffer.getvalue().encode("utf-8")


def format_conversation_for_txt(messages):
    """格式化對話逐字稿"""
    transcript = [f"情緒模式: {st.session_state.emotion_mode}", f"階段: {st.session_state.stage}"]
    transcript.append("=" * 50)
    for msg in messages:
        role = "醫學生" if msg["role"] == "user" else ROLE_LABEL
        transcript.append(f"({role})\n{_strip_visual_tags(msg['content'])}\n")
    return "\n".join(transcript)


# =========================================================
# 側邊欄
# =========================================================
with st.sidebar:
    if st.button("🔙 返回教案選擇", type="secondary", use_container_width=True):
        reset_to_case_selection()
        st.rerun()
    st.markdown(f"### 當前教案")
    st.markdown(f"**{case_info.get('icon', '')} {case_info.get('name', '')}**")
    st.caption(f"角色：{case_info.get('role', '')}")
    
    st.divider()
    
    # 考生指引與報告摘要（移到對話模式上方）
    if selected_case == "npc":
        with st.expander("📘 考生指引摘錄", expanded=False):
            st.markdown(
                "背景：46 歲男性吳忠明，在內視鏡鼻咽部切片檢查後回診確認報告。  \n"
                "任務：向病人說明病情與後續流程，並確保能回應相關提問。  \n"
                "測驗重點：病情說明、情緒處置以及臨床下一步溝通，時間總長 7 分鐘。"
            )
        with st.expander("🧾 病理報告摘要", expanded=False):
            st.markdown(
                "病理診斷：鼻咽部角化鱗狀細胞癌 (keratinizing squamous cell carcinoma)。  \n"
                "備註：報告放置於診間桌面，醫師口頭揭露前病人不會自行確認為癌症。"
            )
    elif selected_case == "abdominal_pain":
        with st.expander("📘 情境說明", expanded=False):
            st.markdown(
                "**場景**：急診室  \n"
                "**病人**：陳志華先生，75 歲，糖尿病導致末期腎臟病，腹膜透析約兩年。  \n"
                "**現況**：因腹痛 8 小時、發燒、血壓低，已在急救室輸液/氧氣。  \n"
                "**您的角色**：長女（主要照顧者），需與醫學生討論病情與治療選項。"
            )
        with st.expander("🧾 抽血檢驗報告", expanded=False):
            for category, items in LAB_DATA.items():
                st.markdown(f"**{category}**")
                for test_name, value in items.items():
                    st.markdown(f"- {test_name}：{value}")
        with st.expander("🖼️ CT 影像", expanded=False):
            for img_name in CT_IMAGES:
                img_path = PROJECT_ROOT / img_name
                if img_path.exists():
                    st.image(str(img_path), use_container_width=True)
                else:
                    st.warning(f"找不到圖片：{img_name}")
        with st.expander("🧾 衛教重點", expanded=False):
            st.markdown(
                "1. 腹膜透析的無菌操作（洗手、環境清潔）  \n"
                "2. 手術與麻醉風險說明  \n"
                "3. 不手術的後果與替代方案  \n"
                "4. 轉院考量與建議"
            )
    
    st.divider()
    
    # 對話模式切換（三種模式）
    st.markdown("### 🎛️ 對話模式")
    mode_options = ["💬 文字模式", "🎙️ 語音輸入", "🎤 即時語音"]
    # 判斷當前模式
    if st.session_state.voice_mode:
        current_mode_idx = 2  # 即時語音模式
    elif st.session_state.voice_input_mode:
        current_mode_idx = 1  # 語音輸入模式
    else:
        current_mode_idx = 0  # 文字模式
    
    selected_mode = st.radio(
        "選擇對話模式",
        mode_options,
        index=current_mode_idx,
        horizontal=True,
        label_visibility="collapsed",
    )
    
    # 處理模式切換
    new_voice_mode = (selected_mode == "🎤 即時語音")
    new_voice_input_mode = (selected_mode == "🎙️ 語音輸入")
    
    if new_voice_mode != st.session_state.voice_mode or new_voice_input_mode != st.session_state.voice_input_mode:
        st.session_state.voice_mode = new_voice_mode
        st.session_state.voice_input_mode = new_voice_input_mode
        # 切換模式時重置計時器（即時語音模式有自己的計時器）
        st.session_state.conversation_started_at = None
        st.session_state.timer_frozen_at = None
        # 清除評分結果
        st.session_state.last_evaluation = None
        st.session_state.last_evaluation_error = None
        st.session_state.steps_feedback = None
        st.session_state.spikes_feedback = None
        st.session_state.shair_feedback = None
        st.session_state.voice_input_text = ""  # 清除語音輸入暫存
        st.rerun()
    
    # 模式說明
    if st.session_state.voice_mode:
        st.info("🎤 即時語音：使用 OpenAI Realtime API 進行即時語音對話")
        # 語音選擇
        voice_options = {
            "shimmer": "Shimmer（女聲，溫和）",
            "alloy": "Alloy（中性）",
            "echo": "Echo（男聲）",
            "coral": "Coral（女聲）",
            "sage": "Sage（中性，沉穩）",
        }
        st.session_state.voice_selected = st.selectbox(
            "AI 語音",
            list(voice_options.keys()),
            format_func=lambda x: voice_options[x],
            index=list(voice_options.keys()).index(st.session_state.voice_selected),
        )
    elif st.session_state.voice_input_mode:
        st.info("🎙️ 語音輸入：說完後可修改文字，按 Enter 送出")

    st.header("⚙️ 功能選單")
    
    # 情緒模式選擇
    emotion_options = list(EMOTION_MODES.keys())
    emotion_labels = [f"{EMOTION_MODES[m].get('emoji', '')} {m}" for m in emotion_options]
    current_idx = emotion_options.index(st.session_state.emotion_mode) if st.session_state.emotion_mode in emotion_options else 0
    selected_label = st.selectbox("情緒模式", emotion_labels, index=current_idx)
    st.session_state.emotion_mode = emotion_options[emotion_labels.index(selected_label)]
    
    # 醫學生等級（僅鼻咽癌教案）
    if selected_case == "npc":
        st.session_state.student_level = st.selectbox(
            "醫學生等級（影響提示語料）",
            options=[3, 4, 5],
            index=[3, 4, 5].index(st.session_state.student_level),
        )
    
    st.info(f"目前溝通階段：**{st.session_state.stage}**")
    
    # 語音模式下不顯示側邊欄計時器（語音介面有自己的倒數計時）
    if not st.session_state.voice_mode:
        # 即時計時器
        render_live_timer(
            st.session_state.conversation_started_at,
            st.session_state.timer_limit_minutes,
            st.session_state.timeout_triggered,
        )
        
        # 計時器設定
        timer_limit = st.slider(
            "對話時間限制（分鐘，0 表示無）",
            min_value=0,
            max_value=40,
            value=st.session_state.timer_limit_minutes,
        )
        if timer_limit != st.session_state.timer_limit_minutes:
            st.session_state.timer_limit_minutes = timer_limit
            st.session_state.timeout_triggered = False
        
        # 時間到自動產生評分
        auto_download = st.checkbox(
            "時間到自動產生評分",
            value=st.session_state.auto_download_on_timeout,
        )
        st.session_state.auto_download_on_timeout = auto_download
    else:
        st.caption("⏱️ 語音模式的時間限制請在對話介面上方設定")
    
    st.divider()
    
    # 重新開始
    if st.button("🔄 重新開始對話", type="primary"):
        st.session_state.messages = []
        st.session_state.stage = STAGES[0]
        st.session_state.last_evaluation = None
        st.session_state.last_evaluation_error = None
        st.session_state.pending_evaluation = False
        st.session_state.diagnosis_disclosed = False
        st.session_state.conversation_started_at = None
        st.session_state.timer_frozen_at = None
        st.session_state.timeout_triggered = False
        st.session_state.logged_this_session = False
        st.session_state.steps_feedback = None
        st.session_state.spikes_feedback = None
        st.session_state.shair_feedback = None
        st.session_state.last_audio_bytes = None
        st.session_state.last_tts_audio = None
        st.rerun()
    
    st.divider()
    
    # 產生評分
    if st.session_state.messages and not st.session_state.last_evaluation:
        if st.button(
            "🧮 產生評分回饋",
            type="secondary",
            disabled=st.session_state.pending_evaluation,
            help="完成問診後可點擊產生評分與回饋。",
            use_container_width=True,
        ):
            request_evaluation()
            if st.session_state.conversation_started_at and not st.session_state.timer_frozen_at:
                st.session_state.timer_frozen_at = time.time()
            st.rerun()
    
    st.divider()
    
    # 管理員模式
    if ADMIN_ACCESS_CODE:
        code_input = st.text_input("管理員代碼", type="password", help="輸入後可顯示進階下載功能")
        st.session_state.admin_mode = bool(code_input) and code_input == ADMIN_ACCESS_CODE
        if code_input and not st.session_state.admin_mode:
            st.caption("❌ 代碼不正確。請再次確認。")
    else:
        st.session_state.admin_mode = st.checkbox(
            "啟用管理員模式",
            value=st.session_state.admin_mode,
            help="未設定代碼時，可手動切換管理員模式。",
        )
    
    if st.session_state.admin_mode:
        st.caption("🛠️ 管理員模式已啟動，可下載完整評分明細。")

# 預先計算時間資訊供計時器與限制檢查使用
elapsed_seconds = get_elapsed_seconds(st.session_state.conversation_started_at)
limit_seconds = st.session_state.timer_limit_minutes * 60 if st.session_state.timer_limit_minutes else 0
if limit_seconds and elapsed_seconds >= limit_seconds and not st.session_state.timeout_triggered:
    st.session_state.timeout_triggered = True
    if st.session_state.auto_download_on_timeout:
        st.session_state.pending_evaluation = True

# =========================================================
# 主介面
# =========================================================
if selected_case == "npc":
    st.title("🩺 鼻咽癌病情告知模擬")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(
            f"""
**👤 病人資訊 (相關病理報告於功能選單查看）**  
姓名：{PATIENT_PERSONA['demographics']['name']}（{PATIENT_PERSONA['demographics']['age']} 歲，{PATIENT_PERSONA['demographics']['gender']}）  
主訴：{', '.join(PATIENT_PERSONA['medical_history']['presenting_symptoms'])}  
家族史：{PATIENT_PERSONA['medical_history']['family_history']}
"""
        )
    with col2:
        emotion_cfg = EMOTION_MODES[st.session_state.emotion_mode]
        st.markdown(
            f"""
**🎭 情緒狀態**  
{emotion_cfg['emoji']} **{st.session_state.emotion_mode}**  
{emotion_cfg['description']}
"""
        )
elif selected_case == "abdominal_pain":
    st.title("🚑 腹痛 - 家屬溝通模擬")
    col1, col2 = st.columns([3, 2])
    with col1:
        demographics = PATIENT_PERSONA['demographics']
        medical = PATIENT_PERSONA['medical_history']
        st.markdown(
            f"""
**👤 病人資訊**  
姓名：{demographics['patient_name']}（{demographics['patient_age']} 歲，{demographics['patient_gender']}）  
主訴：{', '.join(medical['presenting_symptoms'])}  
病史：{medical.get('diagnosis', '')}，{medical.get('treatment', '')}

**👩 您的角色**：{demographics['family_member']}（{demographics['family_relationship']}）
"""
        )
    with col2:
        emotion_cfg = EMOTION_MODES[st.session_state.emotion_mode]
        st.markdown(
            f"""
**🎭 情緒狀態**  
{emotion_cfg['emoji']} **{st.session_state.emotion_mode}**  
{emotion_cfg['description']}
"""
        )

st.divider()

# =========================================================
# 語音模式的即時對話介面
# =========================================================
def get_voice_system_prompt(case_id: str, emotion_mode: str) -> str:
    """語音模式專用的系統提示詞 - 使用與文字模式一致的完整背景"""
    if case_id == "npc":
        return f"""### 角色設定
你是吳忠明，55 歲男性，剛收到鼻咽癌病理報告的病人。你是工程師，已婚，有兩個兒子（20歲、18歲）。
你因為之前回診時鼻塞、耳悶、頸部有腫塊，做了鼻咽部切片檢查，今天回診看報告。
你還不知道自己得了鼻咽癌。

### 家族史（非常重要）
- 本次回診是你自己一個人前來門診，沒有任何家屬陪同在診間。
- 目前直系親屬（太太、小孩、父母）當中沒有人罹患癌症。
- 但你有癌症家族史：你的叔父（爸爸的弟弟）58 歲時因鼻咽癌過世。
- 若醫學生詢問「有沒有癌症家族史」或「家人有沒有得過癌症」，你必須主動提到這位叔父的病史。
- 若醫學生問「現在家人有沒有癌症」，請回答目前家人沒有癌症，但過去有叔父鼻咽癌過世的家族史。
- 【重要】請使用「叔父」這個稱呼，不要說「叔叔」。

### 情緒模式
{emotion_mode}

### 回覆規則
- 【最重要】你是病人，等待醫學生先開口。絕對不要主動先說話，必須等醫學生開始對話。
- 只用繁體中文口語對話，保持情緒模式一致。
- 你是病人，不是醫生，不要使用醫療術語。
- 回答要簡短自然，1-3 句，最多 40 字。
- 如果醫學生提到「癌」、「腫瘤」、「惡性」等字眼，表現出震驚。
- 在醫學生揭露診斷前，聚焦於症狀感受與等待結果的不安。
- 未從醫學生口中聽到檢查結果前，禁止自行揭露或確認已罹癌。
- 適時表達擔心、疑問，可提及家人、經濟負擔。"""
    
    elif case_id == "abdominal_pain":
        return f"""### 角色設定
你是陳志華先生的長女（主要照顧者），現在在急診室。
爸爸 75 歲，糖尿病導致末期腎臟病，腹膜透析約兩年。
現況：爸爸因腹痛 8 小時、發燒、血壓低，已在急救室輸液/氧氣。
媽媽已過世多年，還有一個弟弟正在路上。

### 情緒模式
{emotion_mode}

### 關於轉院的想法（非常重要，請嚴格遵守時機）
- 【時機控制】你一開始不會主動提轉院。只有當醫學生明確說到「手術」「開刀」這類字眼後，你才會在下一個回覆中提出轉院的想法。
- 當你聽到醫學生提到「手術」「開刀」「要開」等字眼後，你的下一句話要表達：「那要不要轉到大醫院？」或「這裡開刀可以嗎？會不會轉去醫學中心比較好？」
- 在聽到「手術」「開刀」之前，你不要主動提轉院。

### 關於腹膜透析的問題（非常重要）
- 當醫學生問你「還有沒有其他問題？」「還有什麼想問的嗎？」或類似問題時，你要問：「如果我爸爸以後病好了，我想幫爸爸做腹膜透析。請問做腹膜透析時要注意哪些？」

### 結束對話的方式（非常重要）
- 當對話接近尾聲，或醫學生表示說明完畢、詢問是否還有問題且你已問過腹膜透析問題後，你要說：「好，等我弟弟來了之後，我們再討論看看。」
- 這句話代表對話的結束。

### 回覆規則
- 【最重要】你是家屬，等待醫學生先開口。絕對不要主動先說話，必須等醫學生開始對話。
- 只用繁體中文口語對話，保持情緒模式一致。
- 你是家屬，不是醫生，不要使用醫療術語。
- 回答要簡短自然，1-3 句，最多 40 字。
- 回答要貼近真實：短句、口語，如「對」「沒有耶」「我不知道」「那現在怎麼辦」。
- 若醫學生提到手術、麻醉風險、不手術後果、轉院等，用家屬視角追問或表達擔心。
- 若醫學生過度保證，依情緒模式做出質疑或不安。
- 適時表達擔心、焦慮，詢問爸爸的狀況。"""
    
    return "你是一位標準化病人/家屬，請用繁體中文回答。"


def create_realtime_voice_html(api_key: str, system_prompt: str, voice: str, role_label: str, time_limit_seconds: int = 0) -> str:
    """建立即時語音對話的 HTML 組件
    
    Args:
        time_limit_seconds: 時間限制（秒），0 表示無限制
    """
    escaped_prompt = system_prompt.replace('`', "'").replace('$', '').replace('\\', '\\\\')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {{ box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
            .container {{ max-width: 100%; padding: 15px; }}
            .status-bar {{
                display: flex; align-items: center; gap: 10px;
                padding: 10px 15px; background: #f0f2f6; border-radius: 10px; margin-bottom: 12px;
            }}
            .status-indicator {{ width: 12px; height: 12px; border-radius: 50%; background: #ccc; }}
            .status-indicator.connected {{ background: #28a745; animation: pulse 2s infinite; }}
            .status-indicator.speaking {{ background: #007bff; animation: pulse 0.5s infinite; }}
            .status-indicator.ai-speaking {{ background: #fd7e14; animation: pulse 0.5s infinite; }}
            @keyframes pulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} 100% {{ opacity: 1; }} }}
            .controls {{ display: flex; gap: 10px; margin-bottom: 12px; }}
            .btn {{
                padding: 12px 24px; border: none; border-radius: 8px; cursor: pointer;
                font-size: 16px; font-weight: 500; transition: all 0.2s;
            }}
            .btn-primary {{ background: #007bff; color: white; }}
            .btn-primary:hover {{ background: #0056b3; }}
            .btn-danger {{ background: #dc3545; color: white; }}
            .btn-danger:hover {{ background: #c82333; }}
            .btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
            .transcript {{
                max-height: 350px; overflow-y: auto; border: 1px solid #ddd;
                border-radius: 10px; padding: 12px; background: #fafafa;
            }}
            .message {{ padding: 8px 12px; margin-bottom: 8px; border-radius: 10px; max-width: 85%; }}
            .message.user {{ background: #007bff; color: white; margin-left: auto; }}
            .message.assistant {{ background: #e9ecef; color: #333; }}
            .message-role {{ font-size: 11px; opacity: 0.7; margin-bottom: 3px; }}
            .timer {{ font-size: 24px; font-weight: bold; color: #333; }}
            #audio-visualizer {{ width: 100%; height: 50px; background: #f8f9fa; border-radius: 8px; margin-bottom: 12px; }}
            .info-box {{ background: #e7f3ff; border: 1px solid #b6d4fe; border-radius: 8px; padding: 10px; margin-bottom: 12px; font-size: 13px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="status-bar">
                <div id="status-indicator" class="status-indicator"></div>
                <span id="status-text">準備就緒</span>
                <span style="margin-left: auto;" class="timer" id="timer">--:--</span>
            </div>
            
            <canvas id="audio-visualizer"></canvas>
            
            <div class="controls">
                <button id="start-btn" class="btn btn-primary" onclick="startConversation()">🎤 開始對話</button>
                <button id="stop-btn" class="btn btn-danger" onclick="stopConversation()" disabled>⏹️ 結束對話</button>
            </div>
            
            <div class="info-box" id="info-box"></div>
            
            <div class="transcript" id="transcript">
                <div style="text-align: center; color: #666; padding: 20px;">對話記錄將顯示在這裡</div>
            </div>
        </div>
        
        <script>
            const API_KEY = '{api_key}';
            const SYSTEM_PROMPT = `{escaped_prompt}`;
            const VOICE = '{voice}';
            const ROLE_LABEL = '{role_label}';
            const TIME_LIMIT_SECONDS = {time_limit_seconds};
            
            let peerConnection = null;
            let dataChannel = null;
            let mediaStream = null;
            let audioContext = null;
            let analyser = null;
            let isRunning = false;
            let startTime = null;
            let timerInterval = null;
            
            let messageSequence = 0;
            let messages = [];
            let pendingUserTranscript = null;
            let pendingAssistantTranscript = null;
            
            const statusIndicator = document.getElementById('status-indicator');
            const statusText = document.getElementById('status-text');
            const startBtn = document.getElementById('start-btn');
            const stopBtn = document.getElementById('stop-btn');
            const transcriptDiv = document.getElementById('transcript');
            const timerDisplay = document.getElementById('timer');
            const infoBox = document.getElementById('info-box');
            const canvas = document.getElementById('audio-visualizer');
            const canvasCtx = canvas.getContext('2d');
            
            // 初始化顯示
            if (TIME_LIMIT_SECONDS > 0) {{
                const mins = String(Math.floor(TIME_LIMIT_SECONDS / 60)).padStart(2, '0');
                const secs = String(TIME_LIMIT_SECONDS % 60).padStart(2, '0');
                timerDisplay.textContent = mins + ':' + secs;
                infoBox.innerHTML = '點擊「開始對話」後允許使用麥克風，對話限時 <b>' + Math.floor(TIME_LIMIT_SECONDS / 60) + ' 分鐘</b>。';
            }} else {{
                timerDisplay.textContent = '00:00';
                infoBox.textContent = '點擊「開始對話」後允許使用麥克風，然後直接說話，AI 會即時回應。';
            }}
            
            // 清除之前的對話記錄
            try {{
                localStorage.removeItem("rt_conversation_data");
            }} catch(e) {{}}
            
            function updateStatus(status, text) {{
                statusIndicator.className = 'status-indicator ' + status;
                statusText.textContent = text;
            }}
            
            function updateTimer() {{
                if (!startTime) return;
                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                
                // 顯示時間
                if (TIME_LIMIT_SECONDS > 0) {{
                    // 有時間限制時顯示倒數
                    const remaining = Math.max(0, TIME_LIMIT_SECONDS - elapsed);
                    const mins = String(Math.floor(remaining / 60)).padStart(2, '0');
                    const secs = String(remaining % 60).padStart(2, '0');
                    timerDisplay.textContent = mins + ':' + secs;
                    
                    // 剩餘 30 秒時變色警告
                    if (remaining <= 30 && remaining > 0) {{
                        timerDisplay.style.color = '#dc3545';
                    }} else if (remaining > 30) {{
                        timerDisplay.style.color = '#333';
                    }}
                    
                    // 時間到自動停止
                    if (remaining <= 0 && isRunning) {{
                        timerDisplay.style.color = '#dc3545';
                        infoBox.innerHTML = '<span style="color: #dc3545; font-weight: bold;">⏰ 時間到！對話自動結束。</span>';
                        stopConversation();
                        return;
                    }}
                }} else {{
                    // 無時間限制時顯示經過時間
                    const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
                    const secs = String(elapsed % 60).padStart(2, '0');
                    timerDisplay.textContent = mins + ':' + secs;
                }}
            }}
            
            function addMessage(role, content, seq) {{
                const newMsg = {{ role: role, content: content, seq: seq }};
                let insertIndex = messages.length;
                for (let i = 0; i < messages.length; i++) {{
                    if (messages[i].seq > seq) {{ insertIndex = i; break; }}
                }}
                messages.splice(insertIndex, 0, newMsg);
                renderMessages();
            }}
            
            function renderMessages() {{
                transcriptDiv.innerHTML = '';
                if (messages.length === 0) {{
                    transcriptDiv.innerHTML = '<div style="text-align: center; color: #666; padding: 20px;">對話記錄將顯示在這裡</div>';
                    return;
                }}
                const sorted = [...messages].sort((a, b) => a.seq - b.seq);
                sorted.forEach(msg => {{
                    const msgDiv = document.createElement('div');
                    msgDiv.className = 'message ' + msg.role;
                    const roleLabel = msg.role === 'user' ? '醫學生' : ROLE_LABEL;
                    msgDiv.innerHTML = '<div class="message-role">' + roleLabel + '</div><div>' + msg.content + '</div>';
                    transcriptDiv.appendChild(msgDiv);
                }});
                transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
            }}
            
            function visualize() {{
                if (!analyser || !isRunning) return;
                const bufferLength = analyser.frequencyBinCount;
                const dataArray = new Uint8Array(bufferLength);
                analyser.getByteFrequencyData(dataArray);
                canvas.width = canvas.offsetWidth;
                canvas.height = 50;
                canvasCtx.fillStyle = '#f8f9fa';
                canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
                const barWidth = (canvas.width / bufferLength) * 2.5;
                let x = 0;
                for (let i = 0; i < bufferLength; i++) {{
                    const barHeight = (dataArray[i] / 255) * canvas.height;
                    canvasCtx.fillStyle = 'rgb(0, 123, ' + Math.min(255, barHeight + 100) + ')';
                    canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
                    x += barWidth + 1;
                }}
                requestAnimationFrame(visualize);
            }}
            
            async function startConversation() {{
                try {{
                    if (!API_KEY || !API_KEY.startsWith('sk-')) {{
                        throw new Error('API Key 格式不正確');
                    }}
                    
                    updateStatus('', '正在連接...');
                    infoBox.textContent = '正在建立連接...';
                    startBtn.disabled = true;
                    
                    messageSequence = 0;
                    messages = [];
                    pendingUserTranscript = null;
                    pendingAssistantTranscript = null;
                    // 清除之前的對話記錄
                    try {{ localStorage.removeItem("rt_conversation_data"); }} catch(e) {{}}
                    renderMessages();
                    
                    const tokenResponse = await fetch('https://api.openai.com/v1/realtime/sessions', {{
                        method: 'POST',
                        headers: {{ 'Authorization': 'Bearer ' + API_KEY, 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ model: 'gpt-4o-realtime-preview-2024-12-17', voice: VOICE }}),
                    }});
                    
                    if (!tokenResponse.ok) throw new Error('無法獲取 session token');
                    
                    const tokenData = await tokenResponse.json();
                    const ephemeralKey = tokenData.client_secret.value;
                    
                    peerConnection = new RTCPeerConnection();
                    
                    const audioEl = document.createElement('audio');
                    audioEl.autoplay = true;
                    peerConnection.ontrack = (e) => {{ audioEl.srcObject = e.streams[0]; }};
                    
                    mediaStream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
                    peerConnection.addTrack(mediaStream.getTracks()[0], mediaStream);
                    
                    audioContext = new AudioContext();
                    const source = audioContext.createMediaStreamSource(mediaStream);
                    analyser = audioContext.createAnalyser();
                    analyser.fftSize = 256;
                    source.connect(analyser);
                    
                    dataChannel = peerConnection.createDataChannel('oai-events');
                    dataChannel.onopen = () => {{
                        dataChannel.send(JSON.stringify({{
                            type: 'session.update',
                            session: {{ 
                                instructions: SYSTEM_PROMPT, 
                                input_audio_transcription: {{ 
                                    model: 'whisper-1',
                                    language: 'zh'
                                }},
                                turn_detection: {{
                                    type: 'server_vad',
                                    threshold: 0.5,
                                    prefix_padding_ms: 500,
                                    silence_duration_ms: 2000
                                }}
                            }}
                        }}));
                    }};
                    
                    dataChannel.onmessage = (e) => handleServerEvent(JSON.parse(e.data));
                    
                    const offer = await peerConnection.createOffer();
                    await peerConnection.setLocalDescription(offer);
                    
                    const sdpResponse = await fetch('https://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17', {{
                        method: 'POST',
                        headers: {{ 'Authorization': 'Bearer ' + ephemeralKey, 'Content-Type': 'application/sdp' }},
                        body: offer.sdp,
                    }});
                    
                    if (!sdpResponse.ok) throw new Error('無法建立 WebRTC 連接');
                    
                    await peerConnection.setRemoteDescription({{ type: 'answer', sdp: await sdpResponse.text() }});
                    
                    isRunning = true;
                    startTime = Date.now();
                    timerInterval = setInterval(updateTimer, 1000);
                    
                    updateStatus('connected', '已連接 - 請開始說話');
                    infoBox.textContent = '連接成功！請直接說話，AI 會即時回應。';
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    
                    visualize();
                    
                }} catch (error) {{
                    console.error('Error:', error);
                    updateStatus('', '連接失敗');
                    infoBox.textContent = '錯誤：' + error.message;
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                }}
            }}
            
            function handleServerEvent(event) {{
                const type = event.type;
                
                if (type === 'input_audio_buffer.speech_started') {{
                    updateStatus('speaking', '您正在說話...');
                    pendingUserTranscript = {{ seq: ++messageSequence }};
                }}
                else if (type === 'input_audio_buffer.speech_stopped') {{
                    updateStatus('connected', '處理中...');
                }}
                else if (type === 'conversation.item.input_audio_transcription.completed') {{
                    const transcript = event.transcript;
                    if (transcript && pendingUserTranscript) {{
                        addMessage('user', transcript, pendingUserTranscript.seq);
                        pendingUserTranscript = null;
                    }} else if (transcript) {{
                        addMessage('user', transcript, ++messageSequence);
                    }}
                }}
                else if (type === 'response.created') {{
                    pendingAssistantTranscript = {{ seq: ++messageSequence }};
                }}
                else if (type === 'response.audio_transcript.done') {{
                    const transcript = event.transcript;
                    if (transcript && pendingAssistantTranscript) {{
                        addMessage('assistant', transcript, pendingAssistantTranscript.seq);
                        pendingAssistantTranscript = null;
                    }} else if (transcript) {{
                        addMessage('assistant', transcript, ++messageSequence);
                    }}
                    updateStatus('connected', '已連接 - 請繼續說話');
                }}
                else if (type === 'response.audio.delta') {{
                    updateStatus('ai-speaking', 'AI 正在回答...');
                }}
                else if (type === 'error') {{
                    console.error('Server error:', event.error);
                    infoBox.textContent = '錯誤：' + (event.error?.message || '未知錯誤');
                }}
            }}
            
            let conversationData = null;
            
            function copyToClipboard() {{
                if (!conversationData) return;
                navigator.clipboard.writeText(conversationData).then(() => {{
                    const copyBtn = document.getElementById('copy-btn');
                    copyBtn.textContent = '✅ 已複製！';
                    copyBtn.style.background = '#10b981';
                    setTimeout(() => {{
                        copyBtn.textContent = '📋 複製對話記錄';
                        copyBtn.style.background = '#3b82f6';
                    }}, 2000);
                }}).catch(err => {{
                    console.error('Failed to copy:', err);
                    alert('複製失敗，請手動選取並複製上方對話記錄');
                }});
            }}
            
            function stopConversation() {{
                isRunning = false;
                if (timerInterval) clearInterval(timerInterval);
                if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
                if (peerConnection) peerConnection.close();
                if (audioContext) audioContext.close();
                
                const duration = startTime ? Math.floor((Date.now() - startTime) / 1000) : 0;
                
                updateStatus('', '對話已結束');
                startBtn.disabled = false;
                stopBtn.disabled = true;
                
                // 按序號排序
                const sortedMessages = [...messages].sort((a, b) => a.seq - b.seq);
                const data = {{
                    messages: sortedMessages,
                    duration: duration,
                    timestamp: new Date().toISOString()
                }};
                conversationData = JSON.stringify(data);
                
                // 顯示複製按鈕和說明
                infoBox.innerHTML = `
                    <div style="text-align: center;">
                        <p style="margin: 0 0 10px 0; font-weight: bold;">✅ 對話已結束</p>
                        <button id="copy-btn" onclick="copyToClipboard()" style="
                            background: #3b82f6;
                            color: white;
                            border: none;
                            padding: 12px 24px;
                            border-radius: 8px;
                            font-size: 16px;
                            cursor: pointer;
                            margin-bottom: 10px;
                        ">📋 複製對話記錄</button>
                        <p style="margin: 0; font-size: 12px; color: #666;">
                            點擊複製後，請貼到下方輸入框中
                        </p>
                    </div>
                `;
                
                // 存到 localStorage（備用）
                try {{
                    localStorage.setItem("rt_conversation_data", conversationData);
                }} catch(e) {{}}
            }}
        </script>
    </body>
    </html>
    """
    return html_content


# =========================================================
# 根據模式顯示不同介面
# =========================================================
if st.session_state.voice_mode:
    # 語音模式介面
    st.markdown("### 🎤 即時語音對話")
    
    # 時間限制設定（在語音介面上方）
    col_time1, col_time2 = st.columns([3, 1])
    with col_time1:
        voice_time_limit = st.slider(
            "⏱️ 對話時間限制（分鐘，0 表示無限制）",
            min_value=0,
            max_value=20,
            value=st.session_state.timer_limit_minutes if st.session_state.timer_limit_minutes <= 20 else 7,
            key="voice_time_limit_slider"
        )
        st.session_state.timer_limit_minutes = voice_time_limit
    with col_time2:
        if voice_time_limit > 0:
            st.metric("限時", f"{voice_time_limit} 分鐘")
        else:
            st.metric("限時", "無限制")
    
    st.info("點擊「開始對話」後允許使用麥克風，直接說話即可。結束後點擊「結束對話」，再按下方按鈕產生評分。")
    
    # 渲染語音組件
    voice_system_prompt = get_voice_system_prompt(selected_case, st.session_state.emotion_mode)
    time_limit_secs = voice_time_limit * 60 if voice_time_limit else 0
    voice_html = create_realtime_voice_html(
        api_key=st.session_state.openai_api_key,
        system_prompt=voice_system_prompt,
        voice=st.session_state.voice_selected,
        role_label=ROLE_LABEL,
        time_limit_seconds=time_limit_secs,
    )
    components.html(voice_html, height=550, scrolling=True)
    
    st.markdown("---")
    
    # 對話記錄輸入區
    st.markdown("### 📋 貼上對話記錄")
    st.caption("對話結束後，點擊上方「📋 複製對話記錄」按鈕，然後貼到下方輸入框")
    
    # 使用兩欄佈局：左邊輸入框，右邊讀取按鈕
    col_input, col_btn = st.columns([4, 1])
    
    with col_input:
        voice_data_input = st.text_input(
            "對話記錄 (JSON)",
            value="",
            key="voice_data_input",
            label_visibility="collapsed",
            placeholder="貼上對話記錄 JSON..."
        )
    
    with col_btn:
        read_btn = st.button("📥 讀取", type="primary", use_container_width=True)
    
    # 點擊讀取按鈕或自動偵測
    if (read_btn or voice_data_input) and not st.session_state.voice_conversation_ended:
        if voice_data_input:
            try:
                conv_data = json.loads(voice_data_input)
                voice_messages = conv_data.get("messages", [])
                voice_duration = conv_data.get("duration", 0)
                if voice_messages:
                    st.session_state.voice_messages = voice_messages
                    st.session_state.voice_duration = voice_duration
                    st.session_state.voice_conversation_ended = True
                    st.rerun()
                elif read_btn:
                    st.warning("⚠️ 對話記錄中沒有訊息")
            except json.JSONDecodeError:
                if read_btn:
                    st.error("❌ JSON 格式錯誤，請確認複製的內容完整")
    
    # 如果對話已結束且有訊息，顯示評分選項
    if st.session_state.voice_conversation_ended and st.session_state.voice_messages:
        st.success(f"✅ 語音對話已結束，共 {len(st.session_state.voice_messages)} 則訊息，時長 {st.session_state.voice_duration // 60} 分 {st.session_state.voice_duration % 60} 秒")
        
        # 顯示對話逐字稿
        with st.expander("📜 查看對話逐字稿", expanded=True):
            for msg in st.session_state.voice_messages:
                role = "醫學生" if msg.get("role") == "user" else ROLE_LABEL
                st.markdown(f"**{role}**: {msg.get('content', '')}")
        
        if not st.session_state.last_evaluation:
            if st.button("📊 產生評分與回饋", type="primary", use_container_width=True, key="voice_eval_btn"):
                # 將語音對話訊息複製到 messages 以便使用現有評分邏輯
                st.session_state.messages = [
                    {"role": m.get("role"), "content": m.get("content", "")}
                    for m in st.session_state.voice_messages
                ]
                st.session_state.pending_evaluation = True
                if st.session_state.conversation_started_at is None:
                    st.session_state.conversation_started_at = time.time() - st.session_state.voice_duration
                st.rerun()
        
        # 重新開始語音對話按鈕
        if st.button("🔄 重新開始語音對話", type="secondary", key="voice_restart_btn"):
            reset_voice_mode()
            st.rerun()

else:
    # 文字模式介面（原有邏輯）
    # 顯示對話歷史
    for msg in st.session_state.messages:
        avatar = "🧑‍⚕️" if msg["role"] == "user" else AVATAR_PATIENT
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"], unsafe_allow_html=True)

# 觸發評分計算
if st.session_state.pending_evaluation:
    if st.session_state.messages:
        with st.spinner("評分與回饋產生中..."):
            try:
                evaluation_result = generate_conversation_evaluation(st.session_state.messages)
                st.session_state.last_evaluation = {
                    "timestamp": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"),
                    "structured": evaluation_result["structured"],
                    "raw_text": evaluation_result["raw_text"],
                }
                st.session_state.last_evaluation_error = None
            except Exception as exc:
                st.session_state.last_evaluation = None
                st.session_state.last_evaluation_error = str(exc)
    st.session_state.pending_evaluation = False

# 顯示評分結果
if st.session_state.last_evaluation_error:
    st.error(f"⚠️ 產生評分時發生錯誤：{st.session_state.last_evaluation_error}")
elif st.session_state.last_evaluation:
    latest_eval = st.session_state.last_evaluation
    structured_eval = latest_eval.get("structured", {})
    overall = structured_eval.get("overall_performance", {}) or {}
    
    st.success(f"✅ 已於 {latest_eval['timestamp']} 完成評分與回饋。")
    
    col_total, col_rating = st.columns(2)
    col_total.metric("項目評分總分", overall.get("total_score", "N/A"))
    
    rating_5 = overall.get("rating_1_to_5", {}) or {}
    r5_score = rating_5.get("score")
    r5_display = "N/A"
    if r5_score is not None:
        try:
            s = int(r5_score)
            mapping = {1: "差", 2: "待加強", 3: "普通", 4: "良好", 5: "優秀"}
            r5_display = f"{s} {mapping.get(s, '')}".strip()
        except:
            r5_display = str(r5_score)
    col_rating.metric("1-5 級整體表現", r5_display)
    
    brief = structured_eval.get("brief_feedback")
    if brief:
        st.info(f"回饋：{brief}")
    
    # 建立評分項目列表
    score_rows = []
    for item in structured_eval.get("evaluation_items", []) or []:
        if not isinstance(item, dict):
            continue
        score_value = item.get("score")
        try:
            score_value = int(score_value) if score_value is not None else None
        except (TypeError, ValueError):
            pass
        score_rows.append({
            "項目": item.get("item", ""),
            "得分": score_value,
            "說明": item.get("detail", ""),
            "評分理由": item.get("rationale", ""),
        })
    
    # 提取亮點與待加強項目
    def extract_score_highlights(rows):
        numeric_rows = [r for r in rows if isinstance(r.get("得分"), int)]
        if not numeric_rows:
            return [], []
        sorted_rows = sorted(numeric_rows, key=lambda r: r.get("得分", 0), reverse=True)
        max_score = sorted_rows[0]["得分"]
        min_score = sorted_rows[-1]["得分"]
        strengths = [r for r in sorted_rows if r.get("得分") == max_score][:3]
        gaps = [r for r in reversed(sorted_rows) if r.get("得分") == min_score][:3]
        return strengths, gaps
    
    strengths, gaps = extract_score_highlights(score_rows)
    
    def _clean_name(n):
        if "." in n:
            parts = n.split(".", 1)
            if parts[0].strip().isdigit():
                return parts[1].strip()
        return n
    
    if strengths:
        st.markdown("**亮點項目**：" + "、".join(_clean_name(r["項目"]) for r in strengths if r.get("項目")))
    else:
        st.markdown("**亮點項目**：尚未顯著亮點")
    
    if gaps:
        st.markdown("**優先改善**：" + "、".join(_clean_name(r["項目"]) for r in gaps if r.get("項目")))
    else:
        st.markdown("**優先改善**：無明顯低分項目")
    
    # 產生對話逐字稿供回饋函式使用
    conversation_text = _format_conversation_for_model(st.session_state.messages)
    
    # 產生 STEPS、SPIKES 和 SHAIR 回饋（只在沒有時產生，避免每次 rerun 重新呼叫 API）
    if st.session_state.steps_feedback is None or st.session_state.spikes_feedback is None or st.session_state.shair_feedback is None:
        with st.spinner("正在產生 STEPS、SPIKES 與 SHAIR 回饋..."):
            steps_feedback = build_steps_feedback(st.session_state.stage, strengths, gaps, conversation_text)
            spikes_feedback = build_spikes_feedback(st.session_state.stage, strengths, gaps, conversation_text)
            shair_feedback = build_shair_feedback(st.session_state.stage, strengths, gaps, conversation_text)
            st.session_state.steps_feedback = steps_feedback
            st.session_state.spikes_feedback = spikes_feedback
            st.session_state.shair_feedback = shair_feedback
    else:
        steps_feedback = st.session_state.steps_feedback
        spikes_feedback = st.session_state.spikes_feedback
        shair_feedback = st.session_state.shair_feedback
    
    # 回饋順序：STEPS → SPIKES → SHAIR
    st.markdown("**STEPS 回饋**：")
    st.write(steps_feedback)
    
    st.markdown("**SPIKES 回饋**：")
    st.write(spikes_feedback)
    
    st.markdown("**SHAIR 回饋**：")
    st.write(shair_feedback)
    
    # 組合使用者資訊
    user_info = {
        "identity": st.session_state.get("user_identity", ""),
        "group": st.session_state.get("user_group", ""),
        "serial": st.session_state.get("user_serial", ""),
    }
    
    # 產生完整報告
    combined_bytes = build_combined_report(
        st.session_state.messages,
        latest_eval,
        st.session_state.stage,
        st.session_state.emotion_mode,
        strengths,
        gaps,
        steps_feedback,
        spikes_feedback,
        shair_feedback,
        case_name=case_info.get('name', ''),
        user_info=user_info,
    )
    
    # 下載按鈕
    # 根據教案產生檔名前綴
    case_prefix = "鼻咽癌" if selected_case == "npc" else "腹痛" if selected_case == "abdominal_pain" else "對話"
    # 加入使用者資訊到檔名
    user_suffix = ""
    if user_info.get("identity") or user_info.get("group") or user_info.get("serial"):
        user_suffix = f"_{user_info.get('identity', '')}_{user_info.get('group', '')}_{user_info.get('serial', '')}"
    st.download_button(
        "📥 下載對話及評分回饋",
        data=combined_bytes,
        file_name=f"{case_prefix}_評分回饋_{datetime.now(TZ).strftime('%Y%m%d_%H%M%S')}{user_suffix}.txt",
        mime="text/plain",
    )
    
    # 自動記錄並上傳到 Google Drive
    if not st.session_state.logged_this_session:
        with st.spinner("正在儲存記錄並上傳到 Google Drive..."):
            try:
                # 建立淨化後的訊息（移除情緒卡片/HTML）供 log 使用
                cleaned_messages = [
                    {**msg, "content": _strip_visual_tags(msg.get("content", ""))}
                    for msg in st.session_state.messages
                ]
                result = session_logger.log_and_upload(
                    messages=cleaned_messages,
                    evaluation=latest_eval,
                    stage=st.session_state.stage,
                    emotion_mode=st.session_state.emotion_mode,
                    student_level=st.session_state.get("student_level", 3),
                    shair_feedback=shair_feedback,
                    conversation_seconds=get_elapsed_seconds(st.session_state.conversation_started_at),
                    diagnosis_disclosed=st.session_state.diagnosis_disclosed,
                    combined_report_bytes=combined_bytes,
                    case_id=selected_case,
                    case_name=case_info.get('name', ''),
                    user_info=user_info,
                )
                st.session_state.logged_this_session = True
                
                if result.get("drive_file_id"):
                    st.success("✅ 記錄已上傳至 Google Drive")
                elif result.get("error_message"):
                    st.warning(f"⚠️ Google Drive 上傳失敗：{result.get('error_message')}")
            except Exception as exc:
                st.warning(f"⚠️ 自動記錄/上傳時發生錯誤：{exc}")
    
    # 管理員明細下載
    if score_rows and st.session_state.admin_mode:
        if pd is not None:
            score_df = pd.DataFrame(score_rows)
        else:
            score_df = None
        
        with st.expander("查看完整項目明細", expanded=False):
            if score_df is not None:
                st.dataframe(score_df, use_container_width=True)
            else:
                st.table(score_rows)
        
        csv_buffer = io.StringIO()
        csv_writer = csv.writer(csv_buffer)
        csv_writer.writerow(["項目", "得分", "說明", "評分理由"])
        for row in score_rows:
            csv_writer.writerow([
                row.get("項目", ""),
                row.get("得分", ""),
                row.get("說明", ""),
                row.get("評分理由", ""),
            ])
        
        # 加上 UTF-8 BOM 避免 Excel 開啟時中文亂碼
        csv_bytes = b'\xef\xbb\xbf' + csv_buffer.getvalue().encode("utf-8")
        st.download_button(
            "📥 下載評分明細 (CSV)",
            data=csv_bytes,
            file_name=f"評分明細_{selected_case}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{user_suffix}.csv",
            mime="text/csv",
        )
    elif score_rows and not st.session_state.admin_mode:
        st.caption("詳細項目僅限管理員查看。")

# =========================================================
# 對話輸入（文字模式 + 語音輸入模式）
# =========================================================
if not st.session_state.voice_mode:
    # 語音輸入模式：使用 Web Speech API 進行語音轉文字
    if st.session_state.voice_input_mode:
        # 語音輸入介面 - 點擊開始/停止 + AI 語音回覆
        st.markdown("---")
        
        # 說明
        st.markdown("### 🎙️ 語音輸入模式")
        st.caption("點擊「開始錄音」說話，再點一次停止。可修改文字後點擊「送出」，AI 會語音回覆。")
        
        # 初始化 voice_input_submitted
        if "voice_input_submitted" not in st.session_state:
            st.session_state.voice_input_submitted = None
        
        # Web Speech API 語音辨識元件（使用 Streamlit 元件回傳值）
        voice_input_html = """
        <style>
            .voice-input-container {
                display: flex;
                flex-direction: column;
                gap: 12px;
                padding: 20px;
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border-radius: 16px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .voice-btn {
                padding: 14px 28px;
                font-size: 16px;
                font-weight: 600;
                border: none;
                border-radius: 30px;
                cursor: pointer;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                user-select: none;
            }
            .voice-btn.record {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
                min-width: 160px;
            }
            .voice-btn.record:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(79, 172, 254, 0.5);
            }
            .voice-btn.record.recording {
                background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
                box-shadow: 0 4px 15px rgba(245, 87, 108, 0.5);
                animation: pulse-btn 1.5s infinite;
            }
            @keyframes pulse-btn {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.03); }
            }
            .voice-btn.submit {
                background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                color: white;
                box-shadow: 0 4px 15px rgba(17, 153, 142, 0.4);
            }
            .voice-btn.submit:hover {
                transform: translateY(-2px);
            }
            .voice-btn.submit:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }
            .voice-btn.clear {
                background: #6c757d;
                color: white;
                padding: 10px 20px;
                font-size: 14px;
            }
            .status-bar {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                padding: 10px;
                border-radius: 8px;
                font-size: 14px;
                transition: all 0.3s ease;
            }
            .status-bar.idle {
                background: #e9ecef;
                color: #6c757d;
            }
            .status-bar.recording {
                background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
                color: white;
            }
            .status-bar.done {
                background: #d4edda;
                color: #155724;
            }
            .status-bar.sending {
                background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                color: white;
            }
            .pulse-dot {
                width: 10px;
                height: 10px;
                background: white;
                border-radius: 50%;
                animation: pulse 1s infinite;
            }
            @keyframes pulse {
                0%, 100% { transform: scale(1); opacity: 1; }
                50% { transform: scale(1.3); opacity: 0.7; }
            }
            .text-editor {
                width: 100%;
                padding: 12px 16px;
                font-size: 16px;
                border: 2px solid #dee2e6;
                border-radius: 12px;
                outline: none;
                transition: border-color 0.2s ease;
                min-height: 60px;
                resize: vertical;
                font-family: inherit;
                box-sizing: border-box;
            }
            .text-editor:focus {
                border-color: #4facfe;
                box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.2);
            }
            .text-editor.has-content {
                border-color: #11998e;
            }
            .btn-row {
                display: flex;
                gap: 10px;
                justify-content: center;
                flex-wrap: wrap;
            }
            .not-supported {
                padding: 15px;
                background: #fff3cd;
                border: 1px solid #ffc107;
                border-radius: 8px;
                color: #856404;
                text-align: center;
            }
        </style>
        
        <div class="voice-input-container" id="voiceContainer">
            <div id="supportedUI">
                <div class="btn-row">
                    <button id="recordBtn" class="voice-btn record" onclick="toggleRecording()">
                        🎙️ 開始錄音
                    </button>
                </div>
                
                <div id="statusBar" class="status-bar idle">
                    <span id="statusText">點擊按鈕開始錄音</span>
                </div>
                
                <textarea id="textEditor" class="text-editor" 
                          placeholder="語音辨識結果會顯示在這裡，您可以修改後點擊「填入下方」..."
                          rows="2"></textarea>
                
                <div class="btn-row">
                    <button id="clearBtn" class="voice-btn clear" onclick="clearText()">
                        🗑️ 清除
                    </button>
                    <button id="submitBtn" class="voice-btn submit" onclick="submitText()" disabled>
                        📋 複製文字
                    </button>
                </div>
            </div>
            
            <div id="notSupportedUI" class="not-supported" style="display: none;">
                ⚠️ 您的瀏覽器不支援語音辨識功能。請使用 Chrome 或 Edge 瀏覽器。
            </div>
        </div>
        
        <script>
            let recognition = null;
            let finalTranscript = '';
            let isRecording = false;
            
            const recordBtn = document.getElementById('recordBtn');
            const statusBar = document.getElementById('statusBar');
            const statusText = document.getElementById('statusText');
            const textEditor = document.getElementById('textEditor');
            const submitBtn = document.getElementById('submitBtn');
            const supportedUI = document.getElementById('supportedUI');
            const notSupportedUI = document.getElementById('notSupportedUI');
            
            // 檢查瀏覽器支援
            if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                recognition = new SpeechRecognition();
                recognition.continuous = true;
                recognition.interimResults = true;
                recognition.lang = 'zh-TW';
                
                recognition.onstart = function() {
                    isRecording = true;
                    recordBtn.classList.add('recording');
                    recordBtn.innerHTML = '⏹️ 停止錄音';
                    statusBar.className = 'status-bar recording';
                    statusText.innerHTML = '<div class="pulse-dot"></div> 正在聆聽... 再點一次停止';
                };
                
                recognition.onresult = function(event) {
                    let interim = '';
                    for (let i = event.resultIndex; i < event.results.length; i++) {
                        if (event.results[i].isFinal) {
                            finalTranscript += event.results[i][0].transcript;
                        } else {
                            interim += event.results[i][0].transcript;
                        }
                    }
                    textEditor.value = finalTranscript + interim;
                    updateEditorState();
                };
                
                recognition.onerror = function(event) {
                    console.error('Speech recognition error:', event.error);
                    if (event.error !== 'aborted' && event.error !== 'no-speech') {
                        statusText.textContent = '辨識錯誤: ' + event.error;
                    }
                };
                
                recognition.onend = function() {
                    isRecording = false;
                    recordBtn.classList.remove('recording');
                    recordBtn.innerHTML = '🎙️ 開始錄音';
                    
                    if (textEditor.value.trim()) {
                        statusBar.className = 'status-bar done';
                        statusText.textContent = '✓ 辨識完成！可修改文字後點擊送出';
                    } else {
                        statusBar.className = 'status-bar idle';
                        statusText.textContent = '點擊按鈕開始錄音';
                    }
                };
            } else {
                supportedUI.style.display = 'none';
                notSupportedUI.style.display = 'block';
            }
            
            function toggleRecording() {
                if (!recognition) return;
                
                if (isRecording) {
                    recognition.stop();
                } else {
                    finalTranscript = textEditor.value;
                    try {
                        recognition.start();
                    } catch (e) {
                        console.log('Recognition error:', e);
                    }
                }
            }
            
            function clearText() {
                textEditor.value = '';
                finalTranscript = '';
                updateEditorState();
                statusBar.className = 'status-bar idle';
                statusText.textContent = '已清除，點擊按鈕開始錄音';
            }
            
            function updateEditorState() {
                const hasContent = textEditor.value.trim().length > 0;
                submitBtn.disabled = !hasContent;
                textEditor.classList.toggle('has-content', hasContent);
            }
            
            function submitText() {
                const text = textEditor.value.trim();
                if (text) {
                    // 更新狀態
                    statusBar.className = 'status-bar sending';
                    statusText.textContent = '⏳ 正在複製文字...';
                    submitBtn.disabled = true;
                    
                    // 複製到剪貼簿
                    navigator.clipboard.writeText(text).then(function() {
                        statusBar.className = 'status-bar done';
                        statusText.innerHTML = '✓ <b>已複製！</b>請點擊下方輸入框，按 <b>Ctrl+V</b> 貼上，再點送出';
                        
                        // 清除上方文字框
                        textEditor.value = '';
                        finalTranscript = '';
                        updateEditorState();
                    }).catch(function(err) {
                        // 備用方案：使用舊式方法
                        const tempTextarea = document.createElement('textarea');
                        tempTextarea.value = text;
                        document.body.appendChild(tempTextarea);
                        tempTextarea.select();
                        document.execCommand('copy');
                        document.body.removeChild(tempTextarea);
                        
                        statusBar.className = 'status-bar done';
                        statusText.innerHTML = '✓ <b>已複製！</b>請點擊下方輸入框，按 <b>Ctrl+V</b> 貼上，再點送出';
                        
                        textEditor.value = '';
                        finalTranscript = '';
                        updateEditorState();
                    });
                }
            }
            
            // 監聯文字編輯
            textEditor.addEventListener('input', updateEditorState);
            
            // 按 Enter 送出（Shift+Enter 換行）
            textEditor.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    if (!submitBtn.disabled) {
                        submitText();
                    }
                }
            });
            
            // 初始化狀態
            updateEditorState();
        </script>
        """
        
        # 使用 components.html 顯示語音辨識介面
        components.html(voice_input_html, height=280)
        
        # 使用 form 來處理提交
        with st.form(key="voice_input_form", clear_on_submit=True):
            voice_text_input = st.text_area(
                "輸入訊息",
                value="",
                placeholder="👆 點擊上方「複製文字」後，在此按 Ctrl+V 貼上，再點「送出訊息」",
                height=100,
                label_visibility="collapsed",
                key="voice_text_area"
            )
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submit_button = st.form_submit_button("✅ 送出訊息", use_container_width=True, type="primary")
        
        if submit_button and voice_text_input.strip():
            prompt = voice_text_input.strip()
        else:
            prompt = None
    else:
        # 純文字模式
        prompt = st.chat_input("請輸入您的問診內容...")

    # 處理輸入
    if prompt:
        is_first_message = st.session_state.conversation_started_at is None
        if is_first_message:
            st.session_state.conversation_started_at = time.time()
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.last_evaluation = None
        st.session_state.last_evaluation_error = None
        # 清除舊的回饋（因為對話內容變了）
        st.session_state.steps_feedback = None
        st.session_state.spikes_feedback = None
        st.session_state.shair_feedback = None
        
        # 語音輸入模式：清除暫存文字
        if st.session_state.voice_input_mode:
            st.session_state.voice_input_text = ""
        
        if detect_diagnosis_disclosure(prompt):
            st.session_state.diagnosis_disclosed = True
        update_stage(prompt)
        
        with st.chat_message("user", avatar="🧑‍⚕️"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar=AVATAR_PATIENT):
            with st.spinner(f"{ROLE_LABEL}思考回覆中..."):
                try:
                    system_prompt = compose_system_prompt(prompt)
                    temperature = EMOTION_MODES[st.session_state.emotion_mode].get("temperature", 0.7)
                    messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages
                    
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=420,
                    )
                    
                    content = response.choices[0].message.content.strip()
                    annotated = annotate_with_intensity(content, st.session_state.emotion_mode)
                    st.markdown(annotated, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": annotated})
                    
                    # 語音輸入模式：使用 TTS 播放 AI 回覆
                    if st.session_state.voice_input_mode:
                        # 移除情緒標注符號和情緒強度標註，只保留純文字
                        clean_text = content
                        for emoji in ["😢", "😰", "😟", "😔", "😠", "😤", "🙁", "😐", "🤔", "💭", "😱", "😡", "😭", "😊", "🙂", "😒"]:
                            clean_text = clean_text.replace(emoji, "")
                        # 移除情緒強度標註和 HTML 標籤（如果有的話）
                        import re
                        clean_text = re.sub(r'【情緒強度：.*?】', '', clean_text)
                        clean_text = re.sub(r'<[^>]+>', '', clean_text)  # 移除 HTML 標籤
                        clean_text = re.sub(r'---+', '', clean_text)  # 移除分隔線
                        clean_text = re.sub(r'\*\*[^*]+\*\*', '', clean_text)  # 移除粗體標記
                        clean_text = clean_text.strip()
                        
                        # 根據教案選擇 TTS 語音
                        # 鼻咽癌教案（吳忠明）：男性聲音 echo
                        # 腹痛教案（長女）：女性聲音 shimmer
                        if selected_case == "npc":
                            tts_voice = "echo"  # 男聲，適合扮演病人吳忠明
                        else:
                            tts_voice = "shimmer"  # 女聲，適合扮演家屬長女
                        
                        # 使用 OpenAI TTS API 生成語音
                        try:
                            tts_response = client.audio.speech.create(
                                model="tts-1",
                                voice=tts_voice,
                                input=clean_text,
                                speed=1.0,
                            )
                            
                            # 將音頻儲存到 session state，rerun 後播放
                            import base64
                            st.session_state.pending_tts_audio = base64.b64encode(tts_response.content).decode('utf-8')
                            
                        except Exception as tts_err:
                            st.caption(f"⚠️ 語音合成失敗：{tts_err}")
                    
                except AuthenticationError:
                    st.error("❌ OpenAI API 金鑰無效或已過期。")
                except Exception as exc:
                    st.error(f"⚠️ 呼叫 OpenAI API 時發生錯誤：{exc}")
        
        # rerun 以更新對話顯示和清除輸入框
        st.rerun()

# 播放待播放的 TTS 音頻（在 rerun 後執行）
if st.session_state.get("pending_tts_audio") and st.session_state.voice_input_mode:
    audio_html = f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{st.session_state.pending_tts_audio}" type="audio/mp3">
    </audio>
    """
    components.html(audio_html, height=0)
    st.session_state.pending_tts_audio = None  # 清除，避免重複播放

st.divider()
st.caption(f"📚 教案：{case_info['name']}")
