from __future__ import annotations

import hashlib
import json
import math
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from openai import OpenAI


@dataclass
class DialogueSegment:
    level: Optional[int]
    stage: str
    speaker: str
    text: str
    emotion_intensity: Optional[int] = None
    source: Optional[str] = None
    embedding: Optional[List[float]] = None
    cache_key: Optional[str] = field(default=None, init=False)

    def set_cache_key(self) -> None:
        base = f"{self.source or 'NA'}::{self.stage}::{self.text}"
        self.cache_key = hashlib.sha1(base.encode("utf-8")).hexdigest()


class PatientContextEngine:
    """Loads teaching scripts and transcripts to build reference context."""

    STAGE_ORDER: Tuple[str, ...] = ("建立關係", "說明解釋", "總結對話")
    STAGE_KEYWORDS: Dict[str, Tuple[str, ...]] = {
        "建立關係": ("最近", "狀況", "哪裡", " symptom", "睡", "先", "你好"),
        "說明解釋": ("檢查", "報告", "結果", "腫瘤", "治療", "癌", "分期", "副作用"),
        "總結對話": ("安排", "追蹤", "回診", "回家", "注意", "謝謝", "聯絡", "支持"),
    }

    def __init__(
        self,
        script_paths: Iterable[Path],
        transcripts_dir: Optional[Path] = None,
        transcript_limit: int = 3,
        transcript_chars: int = 1200,
        embedding_cache_path: Optional[Path] = None,
    ) -> None:
        self._script_paths = [Path(p) for p in script_paths]
        self._transcripts_dir = Path(transcripts_dir) if transcripts_dir else None
        self._transcript_limit = transcript_limit
        self._transcript_chars = transcript_chars
        self._embedding_cache_path = (
            Path(embedding_cache_path)
            if embedding_cache_path
            else Path(__file__).resolve().parent / "context_embeddings.json"
        )

        self.segments: List[DialogueSegment] = []
        for path in self._script_paths:
            if path.exists():
                self.segments.extend(self._parse_script_file(path))

        self.stage_buckets: Dict[Tuple[Optional[int], str], List[DialogueSegment]] = {}
        for seg in self.segments:
            key = (seg.level, seg.stage)
            self.stage_buckets.setdefault(key, []).append(seg)
            general_key = (None, seg.stage)
            self.stage_buckets.setdefault(general_key, []).append(seg)

        self.transcript_samples: List[str] = self._load_transcript_samples()
        self._embedding_cache: Dict[str, List[float]] = {}
        self._load_embedding_cache()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def normalize_stage(self, value: str) -> str:
        for stage in self.STAGE_ORDER:
            if stage in value:
                return stage
        return value if value in self.STAGE_ORDER else self.STAGE_ORDER[0]

    def build_stage_context(
        self,
        level: int,
        stage: str,
        emotion_mode: str,
        top_n: int = 6,
        query_text: Optional[str] = None,
        embedding_client: Optional[OpenAI] = None,
        embedding_model: Optional[str] = None,
    ) -> str:
        stage = self.normalize_stage(stage)
        selected: List[DialogueSegment]

        if query_text and embedding_client and embedding_model:
            selected = self._retrieve_with_embeddings(
                query_text=query_text,
                level=level,
                stage=stage,
                top_n=top_n,
                client=embedding_client,
                model=embedding_model,
            )
        else:
            candidates = [
                seg
                for seg in self.stage_buckets.get((level, stage), [])
                if seg.speaker == "病人"
            ]
            if len(candidates) < top_n:
                general = [
                    seg
                    for seg in self.stage_buckets.get((None, stage), [])
                    if seg.speaker == "病人"
                ]
                for seg in general:
                    if seg not in candidates:
                        candidates.append(seg)
                    if len(candidates) >= top_n:
                        break

            if not candidates:
                return ""

            candidates.sort(
                key=lambda seg: (
                    seg.emotion_intensity is not None,
                    seg.emotion_intensity or 0,
                    len(seg.text),
                ),
                reverse=True,
            )
            selected = candidates[:top_n]

        intensities = [seg.emotion_intensity for seg in selected if seg.emotion_intensity is not None]
        intensity_note = ""
        if intensities:
            avg = sum(intensities) / len(intensities)
            intensity_note = f"平均情緒強度 ≈ {avg:.1f}/10"

        lines = []
        for seg in selected:
            intensity_tag = f"(情緒{seg.emotion_intensity}/10) " if seg.emotion_intensity is not None else ""
            source_tag = f"〔{seg.source}〕" if seg.source else ""
            lines.append(f"- {intensity_tag}{seg.text}{source_tag}")

        header = f"病人語料參考｜階段：{stage}｜Level {level}｜情緒模式：{emotion_mode}"
        parts = [header]
        if intensity_note:
            parts.append(intensity_note)
        parts.extend(lines)
        return "\n".join(parts)

    def build_context_block(
        self,
        level: int,
        stage: str,
        emotion_mode: str,
        query_text: Optional[str] = None,
        embedding_client: Optional[OpenAI] = None,
        embedding_model: Optional[str] = None,
        transcript_chars: Optional[int] = None,
    ) -> str:
        stage_block = self.build_stage_context(
            level=level,
            stage=stage,
            emotion_mode=emotion_mode,
            query_text=query_text,
            embedding_client=embedding_client,
            embedding_model=embedding_model,
        )
        transcript_block = self.sample_transcripts(transcript_chars)
        parts = []
        if stage_block:
            parts.append("#### 教學腳本片段\n" + stage_block)
        if transcript_block:
            parts.append("#### 逐字稿語氣參考\n" + transcript_block)
        return "\n\n".join(parts)

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

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def infer_stage_from_text(text: str, current_stage: str) -> str:
        lowered = text.lower()
        stage_order = list(PatientContextEngine.STAGE_ORDER)
        try:
            current_index = stage_order.index(current_stage)
        except ValueError:
            current_index = 0

        for target_stage in stage_order[current_index:]:
            keywords = PatientContextEngine.STAGE_KEYWORDS.get(target_stage, ())
            if any(keyword in lowered for keyword in keywords):
                return target_stage

        for target_stage in stage_order:
            keywords = PatientContextEngine.STAGE_KEYWORDS.get(target_stage, ())
            if any(keyword in lowered for keyword in keywords):
                return target_stage

        return current_stage

    # ------------------------------------------------------------------
    # Internal parsing routines
    # ------------------------------------------------------------------

    def _parse_script_file(self, path: Path) -> List[DialogueSegment]:
        text = path.read_text(encoding="utf-8")
        segments: List[DialogueSegment] = []
        current_stage = self.STAGE_ORDER[0]
        current_level: Optional[int] = None

        stage_pattern = re.compile(r"(建立關係|說明解釋|總結對話)")
        level_pattern = re.compile(r"Level\s*(\d)")
        dialogue_pattern = re.compile(r"^(醫學生|病人)：(.+)$")
        intensity_pattern = re.compile(r"（[^（）]*?：(\d+)分）")

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            stage_match = stage_pattern.search(line)
            if stage_match:
                current_stage = self.normalize_stage(stage_match.group(1))

            level_match = level_pattern.search(line)
            if level_match:
                try:
                    current_level = int(level_match.group(1))
                except ValueError:
                    current_level = None

            dialogue_match = dialogue_pattern.match(line)
            if not dialogue_match:
                continue

            speaker = dialogue_match.group(1)
            content = dialogue_match.group(2).strip()

            intensity = None
            intensity_match = intensity_pattern.search(content)
            if intensity_match:
                try:
                    intensity = int(intensity_match.group(1))
                except ValueError:
                    intensity = None
                content = intensity_pattern.sub("", content).strip()

            segments.append(
                DialogueSegment(
                    level=current_level,
                    stage=current_stage,
                    speaker=speaker,
                    text=content,
                    emotion_intensity=intensity,
                    source=path.name,
                )
            )
            segments[-1].set_cache_key()

        return segments

    def _load_transcript_samples(self) -> List[str]:
        if not self._transcripts_dir or not self._transcripts_dir.exists():
            return []

        samples: List[str] = []
        for path in sorted(self._transcripts_dir.glob("*.txt"))[: self._transcript_limit]:
            raw_text = path.read_text(encoding="utf-8").strip()
            if not raw_text:
                continue
            collapsed = re.sub(r"\s+", " ", raw_text)
            snippet = collapsed[: self._transcript_chars]
            samples.append(textwrap.dedent(f"〔{path.name} 節錄〕\n{snippet}"))
        return samples

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _load_embedding_cache(self) -> None:
        if not self._embedding_cache_path.exists():
            self._embedding_cache = {}
            return
        try:
            with self._embedding_cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._embedding_cache = {k: v for k, v in data.items() if isinstance(v, list)}
        except Exception:
            self._embedding_cache = {}

        for seg in self.segments:
            if seg.cache_key and seg.cache_key in self._embedding_cache:
                seg.embedding = self._embedding_cache[seg.cache_key]

    def _save_embedding_cache(self) -> None:
        try:
            with self._embedding_cache_path.open("w", encoding="utf-8") as f:
                json.dump(self._embedding_cache, f, ensure_ascii=False)
        except Exception:
            pass

    def _ensure_segment_embeddings(self, client: OpenAI, model: str) -> None:
        missing = [seg for seg in self.segments if seg.embedding is None]
        if not missing:
            return

        batch_size = 64
        for i in range(0, len(missing), batch_size):
            batch = missing[i : i + batch_size]
            texts = [seg.text for seg in batch]
            response = client.embeddings.create(model=model, input=texts)
            for seg, embedding in zip(batch, response.data):
                seg.embedding = embedding.embedding
                if seg.cache_key:
                    self._embedding_cache[seg.cache_key] = seg.embedding

        self._save_embedding_cache()

    def _compute_query_embedding(self, client: OpenAI, model: str, text: str) -> List[float]:
        response = client.embeddings.create(model=model, input=[text])
        return response.data[0].embedding

    def _retrieve_with_embeddings(
        self,
        query_text: str,
        level: int,
        stage: str,
        top_n: int,
        client: OpenAI,
        model: str,
    ) -> List[DialogueSegment]:
        self._ensure_segment_embeddings(client, model)

        candidates = [
            seg
            for seg in self.stage_buckets.get((level, stage), [])
            if seg.speaker == "病人" and seg.embedding is not None
        ]

        if len(candidates) < top_n:
            fallback = [
                seg
                for seg in self.stage_buckets.get((None, stage), [])
                if seg.speaker == "病人" and seg.embedding is not None
            ]
            for seg in fallback:
                if seg not in candidates:
                    candidates.append(seg)
                if len(candidates) >= top_n:
                    break

        if not candidates:
            return []

        query_embedding = self._compute_query_embedding(client, model, query_text)

        def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
            dot = sum(a * b for a, b in zip(vec_a, vec_b))
            norm_a = math.sqrt(sum(a * a for a in vec_a))
            norm_b = math.sqrt(sum(b * b for b in vec_b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        scored = [
            (cosine_similarity(query_embedding, seg.embedding or []), seg)
            for seg in candidates
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        selected = [seg for _, seg in scored[:top_n]]
        if not selected:
            return []

        return selected


__all__ = ["DialogueSegment", "PatientContextEngine"]
