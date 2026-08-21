from __future__ import annotations

import pytest

from em_music_sync import PcmFrame, StreamState, decode, encode_clear, encode_end, encode_pcm, encode_start


def test_round_trip_all_frame_types():
    assert decode(encode_start(7)) == ("start", 7, None)
    assert decode(encode_clear(7)) == ("clear", 7, None)
    assert decode(encode_end(7)) == ("end", 7, None)
    frame = PcmFrame(7, 11, 123456, b"\x01\x00\x02\x00")
    assert decode(encode_pcm(frame)) == ("pcm", 7, frame)


@pytest.mark.parametrize("generation", [0, -1, 2**32])
def test_rejects_invalid_generation(generation):
    with pytest.raises(ValueError):
        encode_start(generation)


def test_rejects_invalid_pcm_boundaries():
    with pytest.raises(ValueError):
        encode_pcm(PcmFrame(1, 0, 0, b""))
    with pytest.raises(ValueError):
        encode_pcm(PcmFrame(1, 0, 0, b"\x00"))
    with pytest.raises(ValueError):
        decode(bytes([0x07]) + b"\x00" * 16 + b"\x01")
    with pytest.raises(ValueError):
        decode(encode_start(1) + b"\x00")


def test_stream_state_rejects_stale_and_duplicate_frames():
    state = StreamState()
    assert state.start(1)
    assert not state.start(1)
    assert not state.start(0)
    assert state.accept(PcmFrame(1, 0, 100, b"\x00\x00"))
    assert not state.accept(PcmFrame(1, 0, 200, b"\x00\x00"))
    assert not state.accept(PcmFrame(1, 0, 300, b"\x00\x00"))
    assert not state.accept(PcmFrame(2, 1, 400, b"\x00\x00"))
    assert state.clear(1)
    assert state.accept(PcmFrame(1, 0, 500, b"\x00\x00"))
    assert state.end(1)
    assert not state.accept(PcmFrame(1, 1, 600, b"\x00\x00"))
