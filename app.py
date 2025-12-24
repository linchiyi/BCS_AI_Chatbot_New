"""
OSCE 醫病對話模擬器 - 多教案整合版
在進入對話前選擇教案，每個教案有獨立的 context engine 確保不會互相影響
"""

import csv
import io
import json
import os
import sys
import time
from datetime import datetime
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
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("PATIENT_MODEL", "gpt-4.1")
EMBEDDING_MODEL = os.getenv("PATIENT_EMBEDDING_MODEL", "text-embedding-3-large")
EVALUATION_MODEL = os.getenv("PATIENT_EVALUATION_MODEL", "gpt-4.1")
ADMIN_ACCESS_CODE = os.getenv("CHATBOT_ADMIN_CODE", "")

PROJECT_ROOT = Path(__file__).resolve().parent

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
# Session State 初始化
# =========================================================
if "selected_case" not in st.session_state:
    st.session_state.selected_case = None
if "case_confirmed" not in st.session_state:
    st.session_state.case_confirmed = False


def reset_to_case_selection():
    """返回教案選擇頁面"""
    st.session_state.selected_case = None
    st.session_state.case_confirmed = False
    # 清除其他對話相關的 session state
    keys_to_clear = [
        "messages", "emotion_mode", "stage", "student_level",
        "last_evaluation", "last_evaluation_error", "pending_evaluation",
        "diagnosis_disclosed", "conversation_started_at", "timer_frozen_at",
        "timeout_triggered", "logged_this_session", "admin_mode",
        "context_engine", "case_config",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


# =========================================================
# 教案選擇頁面
# =========================================================
if not st.session_state.case_confirmed:
    st.title("🏥 OSCE 醫病對話模擬器")
    st.markdown("---")
    st.subheader("請選擇練習教案")
    st.markdown("每個教案有獨立的對話情境和評分標準。選擇後將進入對應的模擬對話。")
    st.markdown("")
    
    # 教案選擇卡片
    cols = st.columns(2)
    
    for idx, (case_id, case_info) in enumerate(CASE_OPTIONS.items()):
        with cols[idx]:
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

# 檢查 API Key
if not API_KEY:
    st.error("❌ 找不到 OPENAI_API_KEY。請建立 .env 並設定金鑰。")
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
try:
    DRIVE_FOLDER_ID = st.secrets.get("DRIVE_FOLDER_ID", os.getenv("GOOGLE_DRIVE_FOLDER_ID", ""))
except:
    DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")

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


def annotate_with_intensity(content: str, emotion_mode: str) -> str:
    if "情緒強度" in content:
        return content
    intensity = EMOTION_MODES.get(emotion_mode, {}).get("intensity", 3)
    return f"{content}\n\n【情緒強度：{emotion_mode} {int(intensity)}/5】"


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
        content = msg.get("content", "").strip()
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
    spikes_feedback: str,
    shair_feedback: str,
    case_name: str = "",
) -> bytes:
    """建立完整的評分報告"""
    buffer = io.StringIO()
    buffer.write("=== 對話概覽 ===\n")
    if case_name:
        buffer.write(f"教案：{case_name}\n")
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
        transcript.append(f"({role})\n{msg['content']}\n")
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
    # st.markdown("---") 
    
    # if st.button("🔙 返回教案選擇", type="secondary", use_container_width=True):
    #     reset_to_case_selection()
    #     st.rerun()

    # st.markdown("---")
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
        st.session_state.spikes_feedback = None
        st.session_state.shair_feedback = None
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
    
    # 考生指引與報告摘要（僅鼻咽癌教案）
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
        with st.expander("🧾 衛教重點", expanded=False):
            st.markdown(
                "1. 腹膜透析的無菌操作（洗手、環境清潔）  \n"
                "2. 手術與麻醉風險說明  \n"
                "3. 不手術的後果與替代方案  \n"
                "4. 轉院考量與建議"
            )
    
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

# 顯示對話歷史
for msg in st.session_state.messages:
    avatar = "🧑‍⚕️" if msg["role"] == "user" else AVATAR_PATIENT
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# 觸發評分計算
if st.session_state.pending_evaluation:
    if st.session_state.messages:
        with st.spinner("評分與回饋產生中..."):
            try:
                evaluation_result = generate_conversation_evaluation(st.session_state.messages)
                st.session_state.last_evaluation = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
    
    # 產生 SPIKES 和 SHAIR 回饋（只在沒有時產生，避免每次 rerun 重新呼叫 API）
    if st.session_state.spikes_feedback is None or st.session_state.shair_feedback is None:
        with st.spinner("正在產生 SPIKES 與 SHAIR 回饋..."):
            spikes_feedback = build_spikes_feedback(st.session_state.stage, strengths, gaps, conversation_text)
            shair_feedback = build_shair_feedback(st.session_state.stage, strengths, gaps, conversation_text)
            st.session_state.spikes_feedback = spikes_feedback
            st.session_state.shair_feedback = shair_feedback
    else:
        spikes_feedback = st.session_state.spikes_feedback
        shair_feedback = st.session_state.shair_feedback
    
    st.markdown("**SPIKES 回饋**：")
    st.write(spikes_feedback)
    
    st.markdown("**SHAIR 回饋**：")
    st.write(shair_feedback)
    
    # 產生完整報告
    combined_bytes = build_combined_report(
        st.session_state.messages,
        latest_eval,
        st.session_state.stage,
        st.session_state.emotion_mode,
        strengths,
        gaps,
        spikes_feedback,
        shair_feedback,
        case_name=case_info.get('name', ''),
    )
    
    # 下載按鈕
    # 根據教案產生檔名前綴
    case_prefix = "鼻咽癌" if selected_case == "npc" else "腹痛" if selected_case == "abdominal_pain" else "對話"
    st.download_button(
        "📥 下載對話及評分回饋",
        data=combined_bytes,
        file_name=f"{case_prefix}_評分回饋_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    )
    
    # 自動記錄並上傳到 Google Drive
    if not st.session_state.logged_this_session:
        with st.spinner("正在儲存記錄並上傳到 Google Drive..."):
            try:
                result = session_logger.log_and_upload(
                    messages=st.session_state.messages,
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
        
        st.download_button(
            "📥 下載評分明細 (CSV)",
            data=csv_buffer.getvalue().encode("utf-8"),
            file_name=f"評分明細_{selected_case}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    elif score_rows and not st.session_state.admin_mode:
        st.caption("詳細項目僅限管理員查看。")

# 對話輸入
prompt = st.chat_input("請輸入您的問診內容...")
if prompt:
    is_first_message = st.session_state.conversation_started_at is None
    if is_first_message:
        st.session_state.conversation_started_at = time.time()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.last_evaluation = None
    st.session_state.last_evaluation_error = None
    # 清除舊的回饋（因為對話內容變了）
    st.session_state.spikes_feedback = None
    st.session_state.shair_feedback = None
    
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
                st.markdown(annotated)
                st.session_state.messages.append({"role": "assistant", "content": annotated})
                
            except AuthenticationError:
                st.error("❌ OpenAI API 金鑰無效或已過期。")
            except Exception as exc:
                st.error(f"⚠️ 呼叫 OpenAI API 時發生錯誤：{exc}")
    
    # 第一則訊息時 rerun，讓側邊欄計時器開始顯示
    if is_first_message:
        st.rerun()

st.divider()
st.caption(f"📚 教案：{case_info['name']}")
