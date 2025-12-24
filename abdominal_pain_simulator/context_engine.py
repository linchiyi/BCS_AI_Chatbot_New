from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

from .srt_parser import SrtCue, load_all_cues


@dataclass
class Utterance:
    transcript_id: str
    start: str
    end: str
    speaker: str  # "家屬" | "考生" | "其他"
    text: str
    embedding: Optional[List[float]] = None
    cache_key: Optional[str] = field(default=None, init=False)

    def set_cache_key(self) -> None:
        base = f"{self.transcript_id}::{self.speaker}::{self.text}"
        self.cache_key = hashlib.sha1(base.encode("utf-8")).hexdigest()


class AbdominalPainContextEngine:
    """Loads real-world transcripts, extracts family utterances, and supports embedding retrieval."""

    DEFAULT_SPEAKER_TARGET = "家屬"

    def __init__(
        self,
        transcripts_dir: Path,
        transcript_limit: int = 5,
        transcript_chars: int = 1600,
        embedding_cache_path: Optional[Path] = None,
        prepared_utterances_path: Optional[Path] = None,
    ) -> None:
        self._transcripts_dir = Path(transcripts_dir)
        self._transcript_limit = transcript_limit
        self._transcript_chars = transcript_chars
        self._embedding_cache_path = (
            Path(embedding_cache_path)
            if embedding_cache_path
            else Path(__file__).resolve().parent / "context_embeddings_abdominal_pain.json"
        )
        self._prepared_utterances_path = Path(prepared_utterances_path) if prepared_utterances_path else None

        self.utterances: List[Utterance] = []
        self.transcript_samples: List[str] = []

        self._embedding_cache: Dict[str, List[float]] = {}
        self._load_embedding_cache()
        self._load_utterances()
        self._load_transcript_samples()

    # -------------------------------
    # Loading
    # -------------------------------

    def _load_embedding_cache(self) -> None:
        if not self._embedding_cache_path.exists():
            self._embedding_cache = {}
            return
        try:
            self._embedding_cache = json.loads(self._embedding_cache_path.read_text(encoding="utf-8"))
        except Exception:
            self._embedding_cache = {}

    def _persist_embedding_cache(self) -> None:
        self._embedding_cache_path.write_text(
            json.dumps(self._embedding_cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_utterances(self) -> None:
        if self._prepared_utterances_path and self._prepared_utterances_path.exists():
            utterances: list[Utterance] = []
            for line in self._prepared_utterances_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                utt = Utterance(
                    transcript_id=str(obj.get("transcript_id", "")),
                    start=str(obj.get("start", "")),
                    end=str(obj.get("end", "")),
                    speaker=str(obj.get("speaker", "其他")),
                    text=str(obj.get("text", "")).strip(),
                    embedding=obj.get("embedding"),
                )
                utt.set_cache_key()
                utterances.append(utt)
            self.utterances = utterances
            return

        # Fallback: heuristic label on SRT cues
        items = load_all_cues(self._transcripts_dir)
        utterances = []
        for transcript_id, cue in items:
            speaker = self._heuristic_label(cue)
            text = cue.text.strip()
            if not text:
                continue
            utt = Utterance(
                transcript_id=transcript_id,
                start=cue.start,
                end=cue.end,
                speaker=speaker,
                text=text,
            )
            utt.set_cache_key()
            utterances.append(utt)
        self.utterances = utterances

    def _load_transcript_samples(self) -> None:
        # Build quick samples comprised of family utterances for style reference.
        by_id: Dict[str, List[str]] = {}
        for utt in self.utterances:
            if utt.speaker != self.DEFAULT_SPEAKER_TARGET:
                continue
            by_id.setdefault(utt.transcript_id, []).append(utt.text)

        samples: list[str] = []
        for transcript_id, texts in sorted(by_id.items()):
            if not texts:
                continue
            snippet = "\n".join(texts[:80]).strip()
            if snippet:
                samples.append(f"〔逐字稿 {transcript_id}｜家屬語氣〕\n{snippet}")
            if len(samples) >= self._transcript_limit:
                break
        self.transcript_samples = samples

    # -------------------------------
    # Public API
    # -------------------------------

    def sample_transcripts(self, total_chars: Optional[int] = None) -> str:
        if not self.transcript_samples:
            return ""
        cap = total_chars or self._transcript_chars
        collected: List[str] = []
        remaining = cap
        for snippet in self.transcript_samples:
            if remaining <= 0:
                break
            collected.append(snippet[:remaining])
            remaining -= len(snippet)
        return "\n---\n".join(collected)

    def retrieve_family_utterances(
        self,
        query_text: str,
        top_n: int,
        client: OpenAI,
        embedding_model: str,
    ) -> List[Utterance]:
        candidates = [u for u in self.utterances if u.speaker == self.DEFAULT_SPEAKER_TARGET]
        if not candidates:
            return []

        query_vec = self._get_embedding(client, embedding_model, "QUERY::" + query_text)
        scored: list[tuple[float, Utterance]] = []
        for utt in candidates:
            vec = utt.embedding
            if vec is None:
                vec = self._get_embedding(client, embedding_model, utt.cache_key or utt.text)
                utt.embedding = vec
            sim = self._cosine_similarity(query_vec, vec)
            scored.append((sim, utt))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [u for _, u in scored[:top_n]]

    def build_context_block(
        self,
        query_text: str,
        client: OpenAI,
        embedding_model: str,
        top_n: int = 10,
        transcript_chars: Optional[int] = None,
    ) -> str:
        retrieved = self.retrieve_family_utterances(
            query_text=query_text,
            top_n=top_n,
            client=client,
            embedding_model=embedding_model,
        )
        lines = [f"- {u.text}〔{u.transcript_id} {u.start}〕" for u in retrieved]
        retrieved_block = "\n".join(lines)
        transcript_block = self.sample_transcripts(transcript_chars)

        parts: list[str] = []
        if retrieved_block:
            parts.append("#### 逐字稿檢索（家屬回覆參考）\n" + retrieved_block)
        if transcript_block:
            parts.append("#### 逐字稿語氣參考\n" + transcript_block)
        return "\n\n".join(parts).strip()

    # -------------------------------
    # Embeddings
    # -------------------------------

    def _get_embedding(self, client: OpenAI, model: str, cache_key: str) -> List[float]:
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        resp = client.embeddings.create(model=model, input=cache_key)
        vec = resp.data[0].embedding
        self._embedding_cache[cache_key] = vec
        # Best-effort persist
        try:
            self._persist_embedding_cache()
        except Exception:
            pass
        return vec

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    # -------------------------------
    # Heuristic labeling fallback
    # -------------------------------

    @staticmethod
    def _heuristic_label(cue: SrtCue) -> str:
        t = cue.text.strip()
        # common clinician phrases
        clinician_markers = (
            "我先自我介紹",
            "我是",
            "請問",
            "那我們",
            "我們現在",
            "我想",
            "我們會",
            "我們先",
            "檢查",
            "報告",
            "電腦斷層",
            "手術",
            "麻醉",
            "抗生素",
            "治療",
            "風險",
        )
        family_markers = (
            "我是他",
            "我弟弟",
            "我哥哥",
            "我爸爸",
            "我媽",
            "轉院",
            "怎麼",
            "為什麼",
            "可以",
            "擔心",
            "對",
            "沒有",
            "不知道",
        )

        if any(m in t for m in clinician_markers):
            return "考生"
        if any(m in t for m in family_markers):
            return "家屬"
        # short replies are more likely family
        if len(t) <= 6:
            return "家屬"
        return "其他"
