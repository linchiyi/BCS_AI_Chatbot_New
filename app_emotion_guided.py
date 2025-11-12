import csv
import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import streamlit as st
from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI

from patient_context_engine import PatientContextEngine

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
MODEL_NAME = os.getenv("PATIENT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("PATIENT_EMBEDDING_MODEL", "text-embedding-3-large")
EVALUATION_MODEL = os.getenv("PATIENT_EVALUATION_MODEL", "gpt-4.1")

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
# 3️⃣ 病患資料與情緒模式
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
    },
}

EMOTION_MODES: Dict[str, Dict[str, str]] = {
    "極度震驚否認型": {
        "emoji": "😱",
        "description": "病人極度震驚，強烈否認診斷，情緒激動",
        "behavior": "- 反覆質疑報告正確性\n- 語無倫次、拒絕接受癌症資訊",
        "temperature": 0.9,
    },
    "恐懼擔憂型": {
        "emoji": "😰",
        "description": "病人接受診斷但極度恐懼，聚焦預後與家人",
        "behavior": "- 反覆詢問存活率與治療副作用\n- 擔心成為家人負擔",
        "temperature": 0.75,
    },
    "冷靜理性型": {
        "emoji": "🤔",
        "description": "病人努力保持冷靜，理性思考治療計畫",
        "behavior": "- 詢問治療流程、費用與成功率\n- 語氣平穩但帶著壓力",
        "temperature": 0.55,
    },
    "悲傷沮喪型": {
        "emoji": "😢",
        "description": "病人極度悲傷，覺得人生失去希望",
        "behavior": "- 常出現無力與自責的語句\n- 需要情緒安撫與陪伴",
        "temperature": 0.65,
    },
    "憤怒質疑型": {
        "emoji": "😠",
        "description": "病人憤怒質疑醫療體系與檢查結果",
        "behavior": "- 語氣強硬，可能指責醫療疏失",
        "temperature": 0.85,
    },
    "接受配合型": {
        "emoji": "💪",
        "description": "病人接受事實，準備積極面對治療",
        "behavior": "- 討論配合事項與生活安排",
        "temperature": 0.6,
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

EVALUATION_SYSTEM_PROMPT = """
你是一位經驗豐富的 OSCE 主考官，負責評估醫學生與標準化病人的完整對話逐字稿。
請依 12 項標準化評分指標進行量化評分，並提供整體回饋。

評分規範：
- 每一項 `score` 必須為 0（未達標）、1（部分達標）、或 2（完全達標）。
- `rating_1_to_5.score` 與 `rating_1_to_3.score` 也需為整數。
- 僅輸出單一 JSON 物件，不得附加說明文字、Markdown 或多餘標點。
- `overall_performance.total_score` 保持為 null，我們會在外部自動計算。
- `brief_feedback` 請提供不超過 40 字的中文重點建議。
- 每一項目請在 `rationale` 欄位以 15 字內說明評分理由。

請使用以下 JSON 模板，並確保鍵名與結構一致：
{
    "evaluation_items": [
        {"item": "1. 有禮貌", "detail": "如聲音態度誠懇，自我介紹，注視病人", "score": null, "rationale": ""},
        {"item": "2. 建立友好關係", "detail": "如稱呼病人姓名及家屬，有需要以外的寒暄語，表達關心或體貼", "score": null, "rationale": ""},
        {"item": "3. 解釋得清楚", "detail": "如說話速度慢，了解病人的相關背景及事前資訊，能就背景給予適切的定對方案建議", "score": null, "rationale": ""},
        {"item": "4. 用心聆聽", "detail": "如眼睛有注視對方，記住對方講的話且有回應，不打斷對方講話", "score": null, "rationale": ""},
        {"item": "5. 同理心", "detail": "如表現出能了解病患感受與處境的語言或態度，適度的回應，提供支持", "score": null, "rationale": ""},
        {"item": "6. 詢問家人是否一起來", "detail": "並告知可請家人一起參與", "score": null, "rationale": ""},
        {"item": "7. 承諾盡心照顧及避免過度的保證", "score": null, "rationale": ""},
        {"item": "8. 以沉默處理沉默及哭泣", "score": null, "rationale": ""},
        {"item": "9. 告知鼻咽癌之預後", "score": null, "rationale": ""},
        {"item": "10. 告知鼻咽癌是否與遺傳相關及相關因子", "score": null, "rationale": ""},
        {"item": "11. 簡要說明鼻咽癌下一步的檢查", "score": null, "rationale": ""},
        {"item": "12. 簡要說明鼻咽癌下一步的治療計畫", "score": null, "rationale": ""}
    ],
    "overall_performance": {
        "total_score": null,
        "rating_1_to_5": {"score": null, "description": "整體表現普通（1=差，2=待加強，3=普通，4=良好，5=優秀）", "reason": ""},
        "rating_1_to_3": {"score": null, "description": "未填寫（1=明顯未達，2=及格基礎，3=明顯通過）", "reason": ""}
    },
    "brief_feedback": ""
}
"""


def _format_conversation_for_model(messages) -> str:
    lines = []
    for idx, message in enumerate(messages, start=1):
        role = "醫學生" if message.get("role") == "user" else "病患"
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
        f"病患情緒模式：{st.session_state.emotion_mode}\n"
        f"對話階段：{st.session_state.stage}\n"
        f"醫學生等級：Level {st.session_state.student_level}\n"
    )

    conversation_text = _format_conversation_for_model(messages)
    user_prompt = f"""
以下提供一段醫學生與標準化病患的完整逐字稿。
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

# =========================================================
# 5️⃣ 工具函式
# =========================================================

def compose_system_prompt(stage: str, latest_user_text: str) -> str:
    emotion_mode = st.session_state.emotion_mode
    level = st.session_state.student_level
    emotion_config = EMOTION_MODES[emotion_mode]

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
    history = PATIENT_PERSONA["medical_history"]

    safeguard = STAGE_SAFEGUARDS.get(stage, "")

    return f"""
### 角色設定
你是 {persona['name']}，{persona['age']} 歲 {persona['gender']}，剛收到鼻咽癌病理報告的病人。醫學生為 Level {level} 學員，正向你說明壞消息。

### 當前溝通階段
- 階段：{stage}
- 指引：{STAGE_GUIDANCE.get(stage, '')}
- 階段安全守則：{safeguard}
- 情緒模式：{emotion_mode}（{emotion_config['description']}）
- 行為特徵：{emotion_config['behavior']}

### 語料參考（請模仿語氣、節奏與字詞選擇）
{context_block}

### 回覆原則
1. 僅以繁體中文回答，保持情緒模式一致。
2. 每次回覆 1-3 句且40字以內為主，必要時可延伸提問或描述身心感受。
4. 若醫學生給出空洞保證，依情緒模式做出相應反應（質疑、恐懼或悲傷）。
5. 適時提出擔心會遺傳給家人，並且適度提及家人、經濟負擔或病友支持，以增加真實感。
3. 不主動提供醫療建議，專注於病人情緒、疑問與生活顧慮。
6. **未從醫學生口中聽到檢查結果、癌症或治療細節前，禁止自行揭露或確認已罹癌；可表達擔心檢查結果，但語氣需保持不確定性。**
""".strip()


def format_conversation_for_txt(messages):
    transcript = [f"情緒模式: {st.session_state.emotion_mode}", f"階段: {st.session_state.stage}"]
    transcript.append("=" * 50)
    for msg in messages:
        role = "醫學生" if msg["role"] == "user" else "病患"
        transcript.append(f"({role})\n{msg['content']}\n")
    return "\n".join(transcript)


def update_stage(user_text: str) -> None:
    current_stage = st.session_state.stage
    inferred = PatientContextEngine.infer_stage_from_text(user_text, current_stage)
    current_index = PatientContextEngine.STAGE_ORDER.index(current_stage)
    inferred_index = PatientContextEngine.STAGE_ORDER.index(inferred)
    if inferred_index > current_index:
        st.session_state.stage = PatientContextEngine.STAGE_ORDER[inferred_index]


# =========================================================
# 6️⃣ 側邊欄
# =========================================================
with st.sidebar:
    st.header("⚙️ 功能選單")

    emotion_options = list(EMOTION_MODES.keys())
    emotion_labels = [f"{EMOTION_MODES[mode]['emoji']} {mode}" for mode in emotion_options]
    current_idx = emotion_options.index(st.session_state.emotion_mode)
    selected_label = st.selectbox("病患情緒模式", emotion_labels, index=current_idx)
    st.session_state.emotion_mode = emotion_options[emotion_labels.index(selected_label)]

    st.session_state.student_level = st.selectbox(
        "醫學生等級（影響提示語料）",
        options=[3, 4, 5],
        index=[3, 4, 5].index(st.session_state.student_level),
    )

    st.info(
        f"目前溝通階段：**{st.session_state.stage}**\n\n"
        f"指引：{STAGE_GUIDANCE.get(st.session_state.stage, '持續觀察病人情緒')}"
    )

    if st.button("🔄 重新開始對話", type="primary"):
        st.session_state.messages = []
        st.session_state.stage = PatientContextEngine.STAGE_ORDER[0]
        st.session_state.last_evaluation = None
        st.session_state.last_evaluation_error = None
        st.session_state.pending_evaluation = False
        st.rerun()

    st.divider()

    if st.session_state.messages:
        transcript_bytes = format_conversation_for_txt(st.session_state.messages).encode("utf-8")
        st.download_button(
            "📥 下載對話紀錄",
            data=transcript_bytes,
            file_name=f"對話紀錄_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            key="download_transcript",
            on_click=request_evaluation,
        )

    # st.caption(f"🧠 使用模型：{MODEL_NAME}")

# =========================================================
# 7️⃣ 主介面
# =========================================================
st.title("🧑‍⚕️ AI 醫病對話 - 語料強化版")

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown(
        f"""
**👤 病患資訊**  
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

for msg in st.session_state.messages:
    avatar = "🧑‍⚕️" if msg["role"] == "user" else "🤒"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# =========================================================
# 8️⃣ 觸發評分計算
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
# 9️⃣ 評分結果顯示
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

    if score_rows:
        st.subheader("項目得分概覽")
        if pd is not None:
            score_df = pd.DataFrame(score_rows)
            chart_series = score_df.set_index("項目")["得分"].fillna(0)
            st.bar_chart(chart_series, use_container_width=True)
            st.dataframe(score_df, use_container_width=True)
        else:
            chart_data = {
                "項目": [row["項目"] for row in score_rows],
                "得分": [row["得分"] if row["得分"] is not None else 0 for row in score_rows],
            }
            st.bar_chart(chart_data, x="項目", y="得分", use_container_width=True)
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
            "📄 下載評分明細 (CSV)",
            data=csv_buffer.getvalue().encode("utf-8"),
            file_name=f"對話評分_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    # with st.expander("查看原始評分 JSON", expanded=False):
    #     st.json(structured_eval)

    # eval_download_data = json.dumps(
    #     structured_eval, ensure_ascii=False, indent=2
    # ).encode("utf-8")
    # st.download_button(
    #     "📊 下載評分 JSON",
    #     data=eval_download_data,
    #     file_name=f"對話評分_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    #     mime="application/json",
    # )

# =========================================================
# 🔟 對話互動
# =========================================================
if prompt := st.chat_input("請輸入您的問診內容..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.last_evaluation = None
    st.session_state.last_evaluation_error = None
    st.session_state.pending_evaluation = False
    update_stage(prompt)

    with st.chat_message("user", avatar="🧑‍⚕️"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤒"):
        with st.spinner("病患思考回覆中..."):
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
                st.markdown(content)
                st.session_state.messages.append({"role": "assistant", "content": content})

            except AuthenticationError:
                st.error("❌ OpenAI API 金鑰無效或已過期。")
            except Exception as exc:
                st.error(f"⚠️ 呼叫 OpenAI API 時發生錯誤：{exc}")

# =========================================================
# 1️⃣1️⃣ 頁尾資訊
# =========================================================
st.divider()
# st.caption(
#     f"階段：{st.session_state.stage} | 情緒模式：{st.session_state.emotion_mode} | 回合：{len(st.session_state.messages)//2}"
# )
