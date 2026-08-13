import wave

import em_debug_audio as audio


def test_debug_audio_is_stored_beside_the_database(tmp_path):
    db_path = str(tmp_path / "echomuse.db")
    assert audio.debug_audio_dir(db_path) == tmp_path / "debug_audio"


def test_debug_audio_names_match_query_contract():
    assert audio.filename(42, "stt") == "42_stt.wav"
    assert audio.filename(42, "loopback") == "42_loopback.wav"
    assert audio.relative_path(42, "stt") == "debug_audio/42_stt.wav"


def test_debug_audio_wav_declares_device_wire_format(tmp_path):
    pcm = b"\x00\x01" * 1600
    path = audio.save(7, "stt", pcm, db_path=str(tmp_path / "echomuse.db"))
    assert path == "debug_audio/7_stt.wav"
    with wave.open(str(tmp_path / path), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (
            16000, 1, 2,
        )
        assert wav.readframes(wav.getnframes()) == pcm


def test_debug_audio_resolve_rejects_untrusted_paths(tmp_path):
    assert audio.resolve("../secret.wav", str(tmp_path / "echomuse.db")) is None
    assert audio.resolve("debug_audio/1_other.wav", str(tmp_path / "echomuse.db")) is None
