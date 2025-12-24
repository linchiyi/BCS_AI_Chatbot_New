from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional


@dataclass(frozen=True)
class SrtCue:
    index: int
    start: str
    end: str
    text: str


_TIME_RANGE_RE = re.compile(r"^(\d\d:\d\d:\d\d,\d\d\d)\s*-->\s*(\d\d:\d\d:\d\d,\d\d\d)\s*$")


def parse_srt_like_text(raw: str) -> List[SrtCue]:
    lines = [ln.rstrip("\n") for ln in raw.splitlines()]
    cues: list[SrtCue] = []
    i = 0

    def next_nonempty(idx: int) -> int:
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        return idx

    while True:
        i = next_nonempty(i)
        if i >= len(lines):
            break

        # index
        try:
            index = int(lines[i].strip())
        except ValueError:
            # If no index, skip this line
            i += 1
            continue
        i += 1

        # time
        if i >= len(lines):
            break
        m = _TIME_RANGE_RE.match(lines[i].strip())
        if not m:
            continue
        start, end = m.group(1), m.group(2)
        i += 1

        # text until blank
        text_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i].strip())
            i += 1

        text = " ".join(text_lines).strip()
        if text:
            cues.append(SrtCue(index=index, start=start, end=end, text=text))

    return cues


def iter_transcript_files(transcripts_dir: Path) -> Iterator[Path]:
    for path in sorted(transcripts_dir.glob("*.txt")):
        if path.name.startswith("."):
            continue
        yield path


def load_all_cues(transcripts_dir: Path) -> List[tuple[str, SrtCue]]:
    items: list[tuple[str, SrtCue]] = []
    for path in iter_transcript_files(transcripts_dir):
        transcript_id = path.stem
        raw = path.read_text(encoding="utf-8", errors="ignore")
        for cue in parse_srt_like_text(raw):
            items.append((transcript_id, cue))
    return items
