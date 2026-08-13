"""Debug microphone audio captured around one voice query."""

from __future__ import annotations

import io
import os
import wave
from pathlib import Path
import re


DEBUG_AUDIO_SUBDIR = "debug_audio"
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1
# Diagnostic captures must stay bounded even if HA never terminates a turn.
MAX_CAPTURE_SECONDS = 180
MAX_CAPTURE_BYTES = MAX_CAPTURE_SECONDS * SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS
_PATH_RE = re.compile(r"^debug_audio/(?P<query>\d+)_(?P<kind>stt|loopback)\.wav$")


def debug_audio_dir(db_path: str | None = None) -> Path:
    if db_path is None:
        db_path = os.environ.get("DB_PATH", "echomuse.db")
    return Path(db_path).resolve().parent / DEBUG_AUDIO_SUBDIR


def filename(query_id: int, kind: str) -> str:
    if kind not in ("stt", "loopback"):
        raise ValueError(f"unknown debug audio kind: {kind}")
    query_id = int(query_id)
    if query_id < 1:
        raise ValueError("query_id must be positive")
    return f"{query_id}_{kind}.wav"


def relative_path(query_id: int, kind: str) -> str:
    return f"{DEBUG_AUDIO_SUBDIR}/{filename(query_id, kind)}"


def encode_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return buf.getvalue()


def save(query_id: int, kind: str, pcm: bytes,
         db_path: str | None = None) -> str | None:
    """Write one immutable WAV and return its DB-relative path."""
    if not pcm:
        return None
    directory = debug_audio_dir(db_path)
    directory.mkdir(parents=True, exist_ok=True)
    name = filename(query_id, kind)
    path = directory / name
    tmp = path.with_suffix(".wav.part")
    tmp.write_bytes(encode_wav(pcm))
    tmp.replace(path)
    return relative_path(query_id, kind)


def resolve(path: str, db_path: str | None = None) -> Path | None:
    """Resolve a stored relative path without allowing traversal."""
    if not isinstance(path, str) or _PATH_RE.fullmatch(path) is None:
        return None
    candidate = debug_audio_dir(db_path) / path.removeprefix(DEBUG_AUDIO_SUBDIR + "/")
    return candidate if candidate.is_file() else None
