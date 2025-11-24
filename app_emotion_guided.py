import csv
import io
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st
from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI
import streamlit.components.v1 as components

from patient_context_engine import PatientContextEngine
from session_logger import SessionLogger

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None

# =========================================================
# 1️⃣ 頁面與樣式設定
# =========================================================
st.set_page_config(
    page_title="AI 醫病對話",
    page_icon="🧑‍⚕️",
    layout="centered",
    initial_sidebar_state="expanded",
)


def load_css(file_name: str) -> None:
    try:
        with open(file_name, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


load_css("styles/main.css")

# =========================================================
# 2️⃣ OpenAI Client 與 Context Engine
# =========================================================
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("PATIENT_MODEL", "gpt-4.1")
EMBEDDING_MODEL = os.getenv("PATIENT_EMBEDDING_MODEL", "text-embedding-3-large")
EVALUATION_MODEL = os.getenv("PATIENT_EVALUATION_MODEL", "gpt-4.1")
ADMIN_ACCESS_CODE = os.getenv("CHATBOT_ADMIN_CODE", "")

if not API_KEY:
    st.error("❌ 找不到 OPENAI_API_KEY。請建立 .env 並設定金鑰。")
    st.stop()

try:
    client = OpenAI(api_key=API_KEY)
except Exception as exc:
    st.error(f"初始化 OpenAI client 失敗：{exc}")
    st.stop()

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SCRIPT_FILES = [
    PROJECT_ROOT.parent / "llm_medical_simulator" / "醫三-五年級的對話腳本_1.txt",
    PROJECT_ROOT.parent / "llm_medical_simulator" / "醫三-五年級的對話腳本_2.txt",
]
DEFAULT_TRANSCRIPTS_DIR = PROJECT_ROOT.parent / "llm_medical_simulator" / "逐字稿_cleaned"


@st.cache_resource(show_spinner=False)
def load_context_engine() -> PatientContextEngine:
    existing_scripts = [path for path in DEFAULT_SCRIPT_FILES if path.exists()]
    return PatientContextEngine(
        script_paths=existing_scripts,
        transcripts_dir=DEFAULT_TRANSCRIPTS_DIR if DEFAULT_TRANSCRIPTS_DIR.exists() else None,
        transcript_limit=4,
        transcript_chars=1600,
    )


context_engine = load_context_engine()

# =========================================================
# 📝 Session Logger 初始化
# =========================================================
DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "16HRRkutsZcscFkk4Q7XgJPEjbz3nurod")
LOGS_DIR = PROJECT_ROOT / "logs"

@st.cache_resource(show_spinner=False)
def get_session_logger() -> SessionLogger:
    return SessionLogger(
        logs_dir=LOGS_DIR,
        drive_folder_id=DRIVE_FOLDER_ID
    )

session_logger = get_session_logger()

# =========================================================
# 3️⃣ 病人資料與情緒模式
# =========================================================
PATIENT_PERSONA = {
    "demographics": {
        "name": "吳宗明",
        "age": 55,
        "gender": "男性",
    },
    "medical_history": {
        "presenting_symptoms": ["持續鼻塞", "有痰", "痰中有血絲"],
        "diagnosis": "鼻咽部角化鱗狀細胞癌",
        "diagnosis_simplified": "鼻咽癌",
        "family_history": "叔父58歲因鼻咽癌過世",
        "children": "兩個兒子 (20歲、18歲)",
        "visit_companions": "本次回診為病人單獨前來，沒有家人陪同。",
    },
}

EMOTION_MODES: Dict[str, Dict[str, str]] = {
    "極度震驚否認型": {
        "emoji": "😱",
        "description": "病人極度震驚，強烈否認診斷，情緒激動",
    "behavior": "- 反覆質疑報告正確性\n- 語無倫次、拒絕接受癌症資訊",
        "temperature": 0.9,
    "intensity": 5,
    },
    "恐懼擔憂型": {
        "emoji": "😰",
        "description": "病人接受診斷但極度恐懼，聚焦預後與家人",
    "behavior": "- 反覆詢問存活率與治療副作用\n- 擔心成為家人負擔",
        "temperature": 0.75,
    "intensity": 4,
    },
    "冷靜理性型": {
        "emoji": "🤔",
        "description": "病人努力保持冷靜，理性思考治療計畫",
    "behavior": "- 詢問治療流程、費用與成功率\n- 語氣平穩但帶著壓力",
        "temperature": 0.55,
    "intensity": 2,
    },
    "悲傷沮喪型": {
        "emoji": "😢",
        "description": "病人極度悲傷，覺得人生失去希望",
    "behavior": "- 常出現無力與自責的語句\n- 需要情緒安撫與陪伴",
        "temperature": 0.65,
    "intensity": 4,
    },
    "憤怒質疑型": {
        "emoji": "😠",
        "description": "病人憤怒質疑醫療體系與檢查結果",
    "behavior": "- 語氣強硬，可能指責醫療疏失",
        "temperature": 0.85,
    "intensity": 5,
    },
    "接受配合型": {
        "emoji": "💪",
        "description": "病人接受事實，準備積極面對治療",
    "behavior": "- 討論配合事項與生活安排",
        "temperature": 0.6,
    "intensity": 3,
    },
}

STAGE_GUIDANCE = {
    "建立關係": "專注於症狀描述、初步情緒反應與對未知的焦慮。",
    "說明解釋": "聚焦於對癌症診斷的震驚、恐懼、資訊需求與家人相關問題。",
    "總結對話": "強調治療安排、生活調整、支持系統與情緒收束。",
}

STAGE_SAFEGUARDS = {
    "建立關係": "醫學生尚未揭露檢查結果時，只能描述症狀、身體不適與不安；不得主動談論癌症、報告或治療。",
    "說明解釋": "若醫學生尚未明確說出癌症或檢查結果，仍須以疑惑或焦慮方式詢問，而非直接表示已被診斷。",
    "總結對話": "仍需遵守前述原則：只有在醫學生已說明癌症後，才可就治療與預後表達擔憂。",
}

PDF_GUIDANCE = {
    "candidate_brief": (
        "背景：46 歲男性吳忠明，在內視鏡鼻咽部切片檢查後回診確認報告。\n"
        "任務：向病人說明病情與後續流程，並確保能回應相關提問。\n"
        "測驗重點：病情說明、情緒處置以及臨床下一步溝通，時間總長 7 分鐘。"
    ),
    "report_summary": (
        "病理診斷：鼻咽部角化鱗狀細胞癌 (keratinizing squamous cell carcinoma)。\n"
        "備註：報告放置於診間桌面，醫師口頭揭露前病人不會自行確認為癌症。"
    ),
}

EVALUATION_SYSTEM_PROMPT = """
你是一位經驗豐富的 OSCE 主考官，負責評估醫學生與標準化病人的完整對話逐字稿。
請依 12 項標準化評分指標進行量化評分，並提供整體回饋。

評分規範：
- 每一項 `score` 必須為 0（未達標）、1（部分達標）、或 2（完全達標）。
- `rating_1_to_5.score` 與 `rating_1_to_3.score` 也需為整數。
- 僅輸出單一 JSON 物件，不得附加說明文字、Markdown 或多餘標點。
- `overall_performance.total_score` 保持為 null，我們會在外部自動計算。
- `brief_feedback` 請提供不超過 40 字的中文重點建議。
- 每一項目請在 `rationale` 欄位以 15 字內說明評分理由（簡要說明為何給此分數）。

請使用以下 JSON 模板，並確保鍵名與結構一致：
{
    "evaluation_items": [
        {
            "item": "1. 有禮貌",
            "detail": "如確定病患姓名、自我介紹、注視病人。完全做到：能做到確定病患姓名、自我介紹、注視病人；部分做到：做到 1-2 項；沒有做到：完全未做到。",
            "score": null,
            "rationale": ""
        },
        {
            "item": "2. 建立友好關係",
            "detail": "如稱呼病人姓名或尊稱、有醫療以外的寒暄話語、表達關心與誠懇。完全做到：三者皆有；部分做到：做到 1-2 項；沒有做到：完全未做到。",
            "score": null,
            "rationale": ""
        },
        {
            "item": "3. 解釋得清楚",
            "detail": "如確認主題、了解病人的相關背景及事前知識、避免專有名詞、確定對方聽懂。完全做到：上述重點多數有做到；部分做到：僅做到其中部分；沒有做到：完全未做到。",
            "score": null,
            "rationale": ""
        },
        {
            "item": "4. 用心聆聽",
            "detail": "如記住對方講的話且有回應、不打斷對方講話。完全做到：能記住對方內容並適當回應且不打斷；部分做到：只做到其中一項；沒有做到：完全未做到。",
            "score": null,
            "rationale": ""
        },
        {
            "item": "5. 同理心",
            "detail": "如表現出能了解病患處境與心境的語言與姿態、適當的語句、提供支持。完全做到：三者皆有；部分做到：做到 1-2 項；沒有做到：完全未做到。",
            "score": null,
            "rationale": ""
        },
        {
            "item": "6. 詢問家人是否一起來，並告知可請家人一起參與",
            "detail": "完全做到：詢問家人是否一起來，並告知可請家人一起參與；部分做到：僅詢問家人是否一起來，未主動邀請家人參與；沒有做到：完全未詢問。",
            "score": null,
            "rationale": ""
        },
        {
            "item": "7. 承諾盡心照顧及避免過度的保證",
            "detail": "完全做到：能明確承諾盡心照顧，同時避免過度或不切實際的保證；部分做到：僅做到其中一項；沒有做到：完全未表達相關內容。",
            "score": null,
            "rationale": ""
        },
        {
            "item": "8. 臨床處理",
            "detail": "請依語意評估醫學生是否有提出實際可行的臨床處理步驟與計畫。完全做到：臨床處理步驟清楚且適切；部分做到：有提到處理方向但不夠完整；沒有做到：未提及具體臨床處理。",
            "score": null,
            "rationale": ""
        },
        {
            "item": "9. 告知鼻咽癌之預後",
            "detail": "說明內容相似即可。完全做到：提到病人五年存活率約 60%，早期病人可高達 90% 以上，而晚期病人也有 50% 以上；部分做到：只說明其中部分內容或大致方向；沒有做到：完全未提到預後。",
            "score": null,
            "rationale": ""
        },
        {
            "item": "10. 告知鼻咽癌是否與遺傳相關",
            "detail": "說明內容相似即可。完全做到：有說明多重因素，例如遺傳因子、EB 病毒感染、環境因素等；部分做到：只說明其中 1-2 項；沒有做到：完全未提到。",
            "score": null,
            "rationale": ""
        },
        {
            "item": "11. 簡要說明鼻咽癌下一步的檢查",
            "detail": "說明內容相似即可。完全做到：指出診斷確立後需先行判定臨床分期，如 CXR、電腦斷層、MRI 等檢查；部分做到：只說明其中一項或方向不完整；沒有做到：完全未提到相關檢查。",
            "score": null,
            "rationale": ""
        },
        {
            "item": "12. 簡要說明鼻咽癌下一步的治療計畫",
            "detail": "說明內容相似即可。完全做到：說明治療主賴放射治療（第一、二期），晚期（第三、四期）或復發病人可能需要併用化學及手術治療；部分做到：只說明其中部分內容或部分正確；沒有做到：完全未提到或內容明顯不正確。",
            "score": null,
            "rationale": ""
        }
    ],
    "overall_performance": {
        "total_score": null,
        "rating_1_to_5": {
            "score": null,
            "description": "整體表現（1=差，2=待加強，3=普通，4=良好，5=優秀）",
            "reason": ""
        },
        "rating_1_to_3": {
            "score": null,
            "description": "整體及格與否（1=明顯未達，2=及格基礎，3=明顯通過）",
            "reason": ""
        }
    },
    "brief_feedback": ""
}
"""


def _format_conversation_for_model(messages) -> str:
    lines = []
    for idx, message in enumerate(messages, start=1):
        role = "醫學生" if message.get("role") == "user" else "病人"
        content = message.get("content", "").strip()
        lines.append(f"{idx}. {role}: {content}")
    return "\n".join(lines)


def _call_evaluation_api(prompt_text: str) -> str:
    try:
        response = client.responses.create(
            model=EVALUATION_MODEL,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": EVALUATION_SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt_text}],
                },
            ],
            temperature=0.0,
        )
    except Exception as exc:
        raise RuntimeError(f"呼叫評分模型失敗：{exc}") from exc

    collected_text: list[str] = []
    output_items = getattr(response, "output", [])
    for item in output_items:
        for content in getattr(item, "content", []):
            if getattr(content, "type", "") in {"output_text", "text"}:
                collected_text.append(getattr(content, "text", ""))

    if not collected_text and hasattr(response, "output_text"):
        collected_text.append(response.output_text)

    raw_text = "\n".join(part for part in collected_text if part).strip()
    if not raw_text:
        raise RuntimeError("評分模型未返回任何文字內容。")
    return raw_text


def _parse_evaluation_output(raw_text: str) -> Dict:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    first = raw_text.find("{")
    last = raw_text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = raw_text[first : last + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError("無法解析評分結果的 JSON。原始輸出：" + raw_text)


def generate_conversation_evaluation(messages) -> Dict:
    if not messages:
        raise ValueError("沒有對話內容可供評分。")

    meta_info = (
        f"病人情緒模式：{st.session_state.emotion_mode}\n"
        f"對話階段：{st.session_state.stage}\n"
        f"醫學生等級：Level {st.session_state.student_level}\n"
    )

    conversation_text = _format_conversation_for_model(messages)
    user_prompt = f"""
以下提供一段醫學生與標準化病人的完整逐字稿。
請依據規範輸出單一 JSON 物件，填寫 12 項評分與整體回饋。
務必遵守分數規範，並於 brief_feedback 中提供 40 字內的中文建議。
如逐字稿有語句不整齊，請依對話語意判斷。

[對話背景]
{meta_info}
[逐字稿]
{conversation_text}
"""

    raw_output = _call_evaluation_api(user_prompt)
    structured = _parse_evaluation_output(raw_output)

    try:
        items = structured.get("evaluation_items", [])
        if isinstance(items, list):
            total = 0
            for item in items:
                if isinstance(item, dict):
                    score = item.get("score")
                    if isinstance(score, (int, float)):
                        item["score"] = int(score)
                        total += int(score)
                    else:
                        item["score"] = 0 if score is None else score
            overall = structured.setdefault("overall_performance", {})
            if isinstance(overall, dict):
                overall["total_score"] = total
    except Exception:
        pass

    return {
        "raw_text": raw_output,
        "structured": structured,
    }


def request_evaluation() -> None:
    st.session_state.pending_evaluation = True

# =========================================================
# 4️⃣ Session State 初始值
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "emotion_mode" not in st.session_state:
    st.session_state.emotion_mode = "恐懼擔憂型"
if "stage" not in st.session_state:
    st.session_state.stage = PatientContextEngine.STAGE_ORDER[0]
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

prompt = st.chat_input("請輸入您的問診內容...")
if prompt and st.session_state.conversation_started_at is None:
    st.session_state.conversation_started_at = time.time()

# =========================================================
# 5️⃣ 工具函式
# =========================================================

def compose_system_prompt(stage: str, latest_user_text: str) -> str:
    emotion_mode = st.session_state.emotion_mode
    level = st.session_state.student_level
    emotion_config = EMOTION_MODES[emotion_mode]
    diagnosis_disclosed = st.session_state.get("diagnosis_disclosed", False)

    context_block = context_engine.build_context_block(
        level=level,
        stage=stage,
        emotion_mode=emotion_mode,
        transcript_chars=1800,
        query_text=latest_user_text,
        embedding_client=client,
        embedding_model=EMBEDDING_MODEL,
    )

    persona = PATIENT_PERSONA["demographics"]
    med_history = PATIENT_PERSONA["medical_history"]
    safeguard = STAGE_SAFEGUARDS.get(stage, "")

    disclosure_note = (
        "醫學生尚未正式告知診斷，病人應維持不確定或焦慮口吻。"
        if not diagnosis_disclosed
        else "醫學生已說明鼻咽癌診斷，可針對治療、預後與家人進一步討論。"
    )

    pre_diagnosis_rules = (
        "- 在醫學生揭露診斷前，回應聚焦於症狀感受、檢查等待的不安與生活受影響處，每次最多提出 1-2 個與症狀或等待相關的問題。\n"
        "- 若醫學生詢問是否知道結果，請表達不安與猜測，但不要自行確認罹癌。"
        if not diagnosis_disclosed
        else "- 可詢問治療、預後與家庭影響，仍需保有原先情緒模式。"
    )

    return f"""
### 角色設定
你是 {persona['name']}，{persona['age']} 歲 {persona['gender']}，剛收到鼻咽癌病理報告的病人。醫學生為 Level {level} 學員，正向你說明壞消息。

- 本次回診是你自己一個人前來門診，沒有任何家屬陪同在診間，回答時不得說太太或家人現在在診間陪同。
- 目前直系親屬（太太、小孩、父母）當中沒有人罹患癌症，但有家族史：{med_history['family_history']}。
- 若醫學生詢問「有沒有癌症家族史」或「家人有沒有得過癌症」，你要主動提到這位叔父的病史。
- 若醫學生問「現在家人有沒有癌症」之類問題，請明確回答目前家人沒有癌症，但過去有叔父鼻咽癌過世的家族史。

### 當前溝通階段
- 階段：{stage}
- 指引：{STAGE_GUIDANCE.get(stage, '')}
- 階段安全守則：{safeguard}
- 情緒模式：{emotion_mode}（{emotion_config['description']}）
- 行為特徵：{emotion_config['behavior']}
- 診斷揭露狀態：{disclosure_note}

### 語料參考（請模仿語氣、節奏與字詞選擇）
{context_block}

### 回覆原則
1. 僅以繁體中文回答，保持情緒模式一致。
2. 每次回覆 1-3 句且 40 字以內為主，必要時可延伸提問或描述身心感受。
3. 不主動提供醫療建議，專注於病人情緒、疑問與生活顧慮。
4. 若醫學生給出空洞保證，依情緒模式做出相應反應（質疑、恐懼或悲傷）。
5. 適時提出擔心會遺傳給家人，並且適度提及家人、經濟負擔或病友支持，以增加真實感。
6. **未從醫學生口中聽到檢查結果、癌症或治療細節前，禁止自行揭露或確認已罹癌；可表達擔心檢查結果，但語氣需保持不確定性。**
7. 每次回覆最後一行，請你自行根據本次回覆的內容與情緒評估強度，加入情緒強度標註，格式固定為：`【情緒強度：<情緒名稱> X/5】`，例如：`【情緒強度：焦慮 4/5】` 或 `【情緒強度：悲傷 5/5】`。

### 情緒與提問節奏
{pre_diagnosis_rules}
""".strip()


def format_conversation_for_txt(messages):
    transcript = [f"情緒模式: {st.session_state.emotion_mode}", f"階段: {st.session_state.stage}"]
    transcript.append("=" * 50)
    for msg in messages:
        role = "醫學生" if msg["role"] == "user" else "病人"
        transcript.append(f"({role})\n{msg['content']}\n")
    return "\n".join(transcript)


def update_stage(user_text: str) -> None:
    current_stage = st.session_state.stage
    inferred = PatientContextEngine.infer_stage_from_text(user_text, current_stage)
    current_index = PatientContextEngine.STAGE_ORDER.index(current_stage)
    inferred_index = PatientContextEngine.STAGE_ORDER.index(inferred)
    if inferred_index > current_index:
        st.session_state.stage = PatientContextEngine.STAGE_ORDER[inferred_index]


DIAGNOSIS_KEY_TERMS = [
    "鼻咽癌",
    "鼻咽部角化鱗狀細胞癌",
    "nasopharyngeal",
    "carcinoma",
    "惡性腫瘤",
]


def detect_diagnosis_disclosure(user_text: str) -> bool:
    """Return True if the醫學生 message揭露了癌症診斷。"""
    text = user_text.strip()
    if not text:
        return False

    lowered = text.lower()
    for term in DIAGNOSIS_KEY_TERMS:
        if term in text or term in lowered:
            return True

    if "癌" in text:
        confirmation_markers = ["確診", "診斷", "報告", "結果", "顯示", "確認", "是", "證實"]
        if any(marker in text for marker in confirmation_markers):
            return True
    return False


def annotate_with_intensity(content: str, emotion_mode: str) -> str:
    """若模型尚未自帶情緒強度標註，才補上一個保守的 1-5 強度。"""
    # 若模型已依照指示輸出【情緒強度：.../5】或含有「情緒強度」字樣，則不再追加
    if "情緒強度" in content:
        return content

    intensity = EMOTION_MODES.get(emotion_mode, {}).get("intensity")
    if intensity is None:
        intensity = 3
    return f"{content}\n\n【情緒強度：{emotion_mode} {int(intensity)}/5】"


def get_elapsed_seconds(start_timestamp: float | None) -> int:
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


def extract_score_highlights(score_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    numeric_rows = [row for row in score_rows if isinstance(row.get("得分"), int)]
    if not numeric_rows:
        return [], []

    sorted_rows = sorted(numeric_rows, key=lambda row: row.get("得分", 0), reverse=True)
    max_score = sorted_rows[0]["得分"]
    min_score = sorted_rows[-1]["得分"]

    strengths = [row for row in sorted_rows if row.get("得分") == max_score][:3]
    focus = [row for row in reversed(sorted_rows) if row.get("得分") == min_score][:3]
    return strengths, focus


def build_shair_feedback(stage: str, strengths: List[Dict[str, Any]], gaps: List[Dict[str, Any]]) -> str:
    def join_items(items: List[Dict[str, Any]]) -> str:
        names = [item.get("項目") for item in items if item.get("項目")]
        return "、".join(names) if names else "尚未顯著項目"

    strength_text = join_items(strengths)
    gap_text = join_items(gaps)

    # 使用評分模型產生約 400-500 字、具 S/H/A/I/R 五段結構的 SHAIR 回饋
    summary_prompt = f"""
你是一位具溝通教學經驗的 OSCE 主考官，熟悉困難溝通中的 SHAIR 模式：
S = Supportive environment（建立支持性的環境與關係）
H = How to deliver（如何傳遞壞消息：語氣、節奏、停頓、用字）
A = Additional information（補充適量且清楚的醫療資訊）
I = Individualize（依病人家庭、身分、價值觀調整說明方式）
R = Reassure and plan（安撫情緒並共同擬定後續計畫）

請根據下列資訊，以 SHAIR 模型對醫學生提供約 400-500 字的中文回饋。

要求：
- 以醫學生為對象，語氣具體、鼓勵且有建設性。
- 依序分成五小段輸出，每一段的開頭請明確以「S (Supportive environment)：」「H (How to deliver)：」「A (Additional information)：」「I (Individualize)：」「R (Reassure and plan)：」標示，後面接上中文說明。
- 每一段內容約 2-4 句完整句子，可使用換行分開段落，但不要使用項目符號或條列清單符號。
- 著重說明本次對話在各面向的優點與可改進處，並提供下次可以實作的 1-2 個具體建議。

[情境階段]
目前溝通階段：{stage}

[亮點項目]
{strength_text}

[優先改善項目]
{gap_text}
""".strip()

    try:
        response = client.responses.create(
            model=EVALUATION_MODEL,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "你是臨床溝通技巧教師，熟悉 SHAIR 模型與 OSCE 評量。",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": summary_prompt}],
                },
            ],
            temperature=0.4,
        )

        collected_text: list[str] = []
        output_items = getattr(response, "output", [])
        for item in output_items:
            for content in getattr(item, "content", []):
                if getattr(content, "type", "") in {"output_text", "text"}:
                    collected_text.append(getattr(content, "text", ""))

        if not collected_text and hasattr(response, "output_text"):
            collected_text.append(response.output_text)

        text = "\n".join(part for part in collected_text if part).strip()
        if text:
            return text
    except Exception:
        # 若生成失敗，退回到簡短版本避免整體流程中斷
        pass

    return (
        f"S（Supportive environment，支持性環境）: 目前對話處於「{stage}」階段，你已經嘗試與病人建立關係並陪伴其面對壞消息，之後可以多留一些時間讓病人表達感受。\n"
        f"H（How to deliver，傳遞方式）: 你在說明 {strength_text} 時的用字與語氣大致穩定，建議在關鍵壞消息或重點句後稍作停頓，觀察病人反應，再繼續補充。\n"
        f"A（Additional information，補充資訊）: 對於 {gap_text} 的解釋還可以更具體、條理化一些，避免一次給太多專有名詞，並適時用生活化例子幫助病人理解。\n"
        f"I（Individualize，個別化）: 回應時可多連結病人的家庭角色與實際處境，像是工作、家中經濟或子女年齡，讓說明更貼近他的擔心。\n"
        f"R（Reassure and plan，安撫與計畫）: 在安撫情緒的同時，簡要說明下一步安排與可利用的資源，讓病人知道不是一個人面對，並對後續有具體方向。"
)


def build_combined_report(
    messages: List[Dict[str, str]],
    evaluation: Dict[str, Any] | None,
    stage: str,
    emotion_mode: str,
    strengths: List[Dict[str, Any]],
    gaps: List[Dict[str, Any]],
    shair_feedback: str,
) -> bytes:
    buffer = io.StringIO()
    buffer.write("=== 對話概覽 ===\n")
    buffer.write(f"階段：{stage}\n")
    buffer.write(f"情緒模式：{emotion_mode}\n")
    # 加入對話總時長
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
        buffer.write(f"總分：{overall.get('total_score', 'N/A')}\n")
        rating_5 = overall.get("rating_1_to_5", {}) or {}
        rating_3 = overall.get("rating_1_to_3", {}) or {}
        buffer.write(f"1-5 級整體評分：{rating_5.get('score', 'N/A')}\n")
        buffer.write(f"1-3 級整體評分：{rating_3.get('score', 'N/A')}\n")
        buffer.write(f"重點回饋：{structured.get('brief_feedback', '')}\n\n")

        buffer.write("=== 亮點項目 ===\n")
        if strengths:
            for item in strengths:
                buffer.write(f"- {item.get('項目')}: {item.get('說明')} (得分 {item.get('得分')})\n")
        else:
            buffer.write("- 尚未顯著亮點\n")

        buffer.write("\n=== 待加強項目 ===\n")
        if gaps:
            for item in gaps:
                buffer.write(f"- {item.get('項目')}: {item.get('說明')} (得分 {item.get('得分')})\n")
        else:
            buffer.write("- 無明顯低分項目\n")

        buffer.write("\n=== SHAIR 回饋 ===\n")
        buffer.write(shair_feedback)
        buffer.write("\n")

    return buffer.getvalue().encode("utf-8")


# 預先計算時間資訊供計時器與限制檢查使用
elapsed_seconds = get_elapsed_seconds(st.session_state.conversation_started_at)

# =========================================================
# 6️⃣ 側邊欄
# =========================================================
with st.sidebar:
    st.header("⚙️ 功能選單")

    emotion_options = list(EMOTION_MODES.keys())
    emotion_labels = [f"{EMOTION_MODES[mode]['emoji']} {mode}" for mode in emotion_options]
    current_idx = emotion_options.index(st.session_state.emotion_mode)
    selected_label = st.selectbox("病人情緒模式", emotion_labels, index=current_idx)
    st.session_state.emotion_mode = emotion_options[emotion_labels.index(selected_label)]

    st.session_state.student_level = st.selectbox(
        "醫學生等級（影響提示語料）",
        options=[3, 4, 5],
        index=[3, 4, 5].index(st.session_state.student_level),
    )

    st.info(
        f"目前溝通階段：**{st.session_state.stage}**\n\n"
        # f"指引：{STAGE_GUIDANCE.get(st.session_state.stage, '持續觀察病人情緒')}"
    )

    # if st.session_state.diagnosis_disclosed:
    #     st.caption("✅ 醫學生已揭露鼻咽癌診斷，可討論治療與家人安排。")
    # else:
    #     st.caption("⏳ 尚未揭露診斷，病患僅能表達症狀與等待報告的不安。")

    render_live_timer(
        st.session_state.conversation_started_at,
        st.session_state.timer_limit_minutes,
        st.session_state.timeout_triggered,
    )

    timer_limit = st.slider(
        "對話時間限制（分鐘，0 表示無）",
        min_value=0,
        max_value=40,
        value=st.session_state.timer_limit_minutes,
    )
    if timer_limit != st.session_state.timer_limit_minutes:
        st.session_state.timer_limit_minutes = timer_limit
        st.session_state.timeout_triggered = False

    auto_download = st.checkbox(
        "時間到自動產生評分提醒",
        value=st.session_state.auto_download_on_timeout,
    )
    st.session_state.auto_download_on_timeout = auto_download

    if st.button("🔄 重新開始對話", type="primary"):
        st.session_state.messages = []
        st.session_state.stage = PatientContextEngine.STAGE_ORDER[0]
        st.session_state.last_evaluation = None
        st.session_state.last_evaluation_error = None
        st.session_state.pending_evaluation = False
        st.session_state.diagnosis_disclosed = False
        st.session_state.conversation_started_at = None
        st.session_state.timer_frozen_at = None
        st.session_state.timeout_triggered = False
        st.session_state.logged_this_session = False
        st.rerun()

    st.divider()

    # 產生評分按鈕
    if st.session_state.messages and not st.session_state.last_evaluation:
        if st.button(
            "🧮 產生評分回饋",
            type="secondary",
            disabled=st.session_state.pending_evaluation,
            help="完成問診後可點擊產生評分與回饋。",
            key="eval_button_sidebar",
            use_container_width=True,
        ):
            request_evaluation()
            # 凍結計時器在按下評分當下的時間
            if st.session_state.conversation_started_at and not st.session_state.timer_frozen_at:
                st.session_state.timer_frozen_at = time.time()
            st.rerun()

    # st.divider()

    with st.expander("📘 考生指引摘錄", expanded=False):
        st.markdown(PDF_GUIDANCE["candidate_brief"].replace("\n", "  \n"))

    with st.expander("🧾 病理報告摘要", expanded=False):
        st.markdown(PDF_GUIDANCE["report_summary"].replace("\n", "  \n"))

    st.divider()

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


limit_seconds = st.session_state.timer_limit_minutes * 60 if st.session_state.timer_limit_minutes else 0
if limit_seconds and elapsed_seconds >= limit_seconds and not st.session_state.timeout_triggered:
    st.session_state.timeout_triggered = True
    st.session_state.pending_evaluation = True

    # st.caption(f"🧠 使用模型：{MODEL_NAME}")

# =========================================================
# 7️⃣ 主介面
# =========================================================
st.title("🧑‍⚕️ AI 醫病對話 - 語料強化版")

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

st.divider()

if st.session_state.timeout_triggered:
    st.warning("⏰ 對話時間已到，請整理重點並下載對話與評分回饋。")

for msg in st.session_state.messages:
    avatar = "🧑‍⚕️" if msg["role"] == "user" else "🤒"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# =========================================================
# 9️⃣ 觸發評分計算
# =========================================================
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
    else:
        st.session_state.last_evaluation = None
        st.session_state.last_evaluation_error = "沒有可評分的對話。"
    st.session_state.pending_evaluation = False

# =========================================================
# 🔟 評分結果顯示
# =========================================================
if st.session_state.last_evaluation_error:
    st.error(f"⚠️ 產生評分時發生錯誤：{st.session_state.last_evaluation_error}")
elif st.session_state.last_evaluation:
    latest_eval = st.session_state.last_evaluation
    structured_eval = latest_eval.get("structured", {})
    overall = structured_eval.get("overall_performance", {}) or {}
    rating_1_to_5 = overall.get("rating_1_to_5", {}) or {}
    rating_1_to_3 = overall.get("rating_1_to_3", {}) or {}

    st.success(f"✅ 已於 {latest_eval['timestamp']} 完成評分與回饋。")

    col_total, col_rating5, col_rating3 = st.columns(3)
    total_score = overall.get("total_score")
    col_total.metric("總分", total_score if total_score is not None else "N/A")
    col_rating5.metric(
        "1-5 級整體評分",
        rating_1_to_5.get("score", "N/A"),
        help=rating_1_to_5.get("description", ""),
    )
    col_rating3.metric(
        "1-3 級整體評分",
        rating_1_to_3.get("score", "N/A"),
        help=rating_1_to_3.get("description", ""),
    )

    brief_feedback = structured_eval.get("brief_feedback")
    if brief_feedback:
        st.info(f"回饋：{brief_feedback}")

    score_rows = []
    for item in structured_eval.get("evaluation_items", []) or []:
        if not isinstance(item, dict):
            continue
        score_value = item.get("score")
        try:
            score_value = int(score_value) if score_value is not None else None
        except (TypeError, ValueError):
            pass
        score_rows.append(
            {
                "項目": item.get("item", ""),
                "得分": score_value,
                "說明": item.get("detail", ""),
                "評分理由": item.get("rationale", ""),
            }
        )

    strengths, gaps = extract_score_highlights(score_rows)
    shair_feedback = build_shair_feedback(st.session_state.stage, strengths, gaps)

    if strengths:
        st.markdown(
            "**亮點項目**：" + "、".join(row["項目"] for row in strengths if row.get("項目"))
        )
    else:
        st.markdown("**亮點項目**：尚未顯著亮點")

    if gaps:
        st.markdown(
            "**優先改善**：" + "、".join(row["項目"] for row in gaps if row.get("項目"))
        )
    else:
        st.markdown("**優先改善**：無明顯低分項目")

    st.markdown("**SHAIR 回饋**：")
    st.write(shair_feedback)

    combined_bytes = build_combined_report(
        st.session_state.messages,
        latest_eval,
        st.session_state.stage,
        st.session_state.emotion_mode,
        strengths,
        gaps,
        shair_feedback,
    )

    st.download_button(
        "📥 下載對話及評分回饋",
        data=combined_bytes,
        file_name=f"對話與評分_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    )

    # 自動記錄並上傳到 Google Drive
    if "logged_this_session" not in st.session_state:
        st.session_state.logged_this_session = False
    
    if not st.session_state.logged_this_session:
        with st.spinner("正在儲存記錄並上傳到 Google Drive..."):
            try:
                result = session_logger.log_and_upload(
                    messages=st.session_state.messages,
                    evaluation=latest_eval,
                    stage=st.session_state.stage,
                    emotion_mode=st.session_state.emotion_mode,
                    student_level=st.session_state.student_level,
                    shair_feedback=shair_feedback,
                    conversation_seconds=get_elapsed_seconds(st.session_state.conversation_started_at),
                    diagnosis_disclosed=st.session_state.diagnosis_disclosed,
                    combined_report_bytes=combined_bytes,
                )
                st.session_state.logged_this_session = True
                
                if result["local_path"]:
                    st.success(f"✅ 記錄已儲存至後端")
                if result["drive_file_id"]:
                    st.success(f"✅ 記錄已上傳至 Google Drive")
                if result["report_drive_id"]:
                    st.success(f"✅ 評分報告已上傳至 Google Drive")
                    
            except Exception as exc:
                st.warning(f"⚠️ 自動記錄/上傳時發生錯誤：{exc}")

    if score_rows:
        if st.session_state.admin_mode:
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
                file_name=f"對話評分_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.caption("詳細項目僅限管理員查看。")


# =========================================================
# 1️⃣1️⃣ 對話互動
# =========================================================
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.last_evaluation = None
    st.session_state.last_evaluation_error = None
    if not st.session_state.timeout_triggered:
        st.session_state.pending_evaluation = False
    if detect_diagnosis_disclosure(prompt):
        st.session_state.diagnosis_disclosed = True
    update_stage(prompt)

    with st.chat_message("user", avatar="🧑‍⚕️"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤒"):
        with st.spinner("病人思考回覆中..."):
            try:
                system_prompt = compose_system_prompt(st.session_state.stage, prompt)
                temperature = EMOTION_MODES[st.session_state.emotion_mode]["temperature"]
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

# =========================================================
# 1️⃣2️⃣ 頁尾資訊
# =========================================================
st.divider()

