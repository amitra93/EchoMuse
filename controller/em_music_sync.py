"""EchoMuse timestamped music data-plane frames.

This is the controller/device transport envelope, not the Sendspin protocol
itself. Sendspin timestamps are converted to device monotonic time before a
PCM frame is encoded here.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct


FRAME_START = 0x06
FRAME_PCM = 0x07
FRAME_CLEAR = 0x08
FRAME_END = 0x09
_GENERATION = struct.Struct(">BI")
_PCM_HEADER = struct.Struct(">BIIq")


@dataclass(frozen=True)
class PcmFrame:
    generation: int
    sequence: int
    target_us: int
    pcm: bytes


def _check_generation(generation: int) -> None:
    if not isinstance(generation, int) or isinstance(generation, bool):
        raise ValueError("generation must be an integer")
    if not 0 < generation <= 0xFFFFFFFF:
        raise ValueError("generation must be between 1 and 2^32-1")


def encode_start(generation: int) -> bytes:
    _check_generation(generation)
    return _GENERATION.pack(FRAME_START, generation)


def encode_clear(generation: int) -> bytes:
    _check_generation(generation)
    return _GENERATION.pack(FRAME_CLEAR, generation)


def encode_end(generation: int) -> bytes:
    _check_generation(generation)
    return _GENERATION.pack(FRAME_END, generation)


def encode_pcm(frame: PcmFrame) -> bytes:
    _check_generation(frame.generation)
    if not 0 <= frame.sequence <= 0xFFFFFFFF:
        raise ValueError("sequence must be between 0 and 2^32-1")
    if not -(1 << 63) <= frame.target_us < (1 << 63):
        raise ValueError("target timestamp is outside int64")
    if not frame.pcm or len(frame.pcm) % 2:
        raise ValueError("PCM must contain at least one complete S16 sample")
    return _PCM_HEADER.pack(FRAME_PCM, frame.generation, frame.sequence, frame.target_us) + frame.pcm


def decode(data: bytes) -> tuple[str, int, PcmFrame | None]:
    """Decode one frame as ``(kind, generation, pcm_frame_or_none)``."""
    if not data:
        raise ValueError("empty music-sync frame")
    kind = data[0]
    if kind in (FRAME_START, FRAME_CLEAR, FRAME_END):
        if len(data) != _GENERATION.size:
            raise ValueError("invalid music-sync control frame length")
        generation = _GENERATION.unpack(data)[1]
        _check_generation(generation)
        return {FRAME_START: "start", FRAME_CLEAR: "clear", FRAME_END: "end"}[kind], generation, None
    if kind != FRAME_PCM or len(data) <= _PCM_HEADER.size:
        raise ValueError("invalid music-sync PCM frame")
    _, generation, sequence, target_us = _PCM_HEADER.unpack(data[:_PCM_HEADER.size])
    frame = PcmFrame(generation, sequence, target_us, data[_PCM_HEADER.size:])
    if len(frame.pcm) % 2:
        raise ValueError("PCM must contain complete S16 samples")
    return "pcm", generation, frame


class StreamState:
    """Reject stale generations and duplicate/out-of-order PCM frames."""

    def __init__(self) -> None:
        self.generation: int | None = None
        self.active = False
        self.last_sequence: int | None = None

    def start(self, generation: int) -> bool:
        try:
            _check_generation(generation)
        except ValueError:
            return False
        if self.generation is not None and generation <= self.generation:
            return False
        self.generation = generation
        self.active = True
        self.last_sequence = None
        return True

    def clear(self, generation: int) -> bool:
        if not self.active or generation != self.generation:
            return False
        self.last_sequence = None
        return True

    def accept(self, frame: PcmFrame) -> bool:
        if not self.active or frame.generation != self.generation:
            return False
        if self.last_sequence is not None and frame.sequence <= self.last_sequence:
            return False
        self.last_sequence = frame.sequence
        return True

    def end(self, generation: int) -> bool:
        if not self.active or generation != self.generation:
            return False
        self.active = False
        self.last_sequence = None
        return True
