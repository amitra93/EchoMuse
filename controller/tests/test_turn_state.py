from em_turn_state import InvalidTransition, TurnEvent, TurnState, transition
from em_turn_state import barge_playback_fired


def test_playback_barge_requires_consecutive_frames():
    assert not barge_playback_fired(0.12, 0.10, 0.0)
    assert barge_playback_fired(0.12, 0.10, 0.11)
    assert not barge_playback_fired(0.12, 0.10, 0.09)

def test_endpoint_events_close_uplink_without_starting_playback():
    assert transition(TurnState.UPLINKING, TurnEvent.HA_VAD_END) is TurnState.WAITING_FOR_TTS
    assert transition(TurnState.UPLINKING, TurnEvent.TTS_URL) is TurnState.WAITING_FOR_TTS


def test_url_starts_playback_after_handoff():
    assert transition(TurnState.WAITING_FOR_TTS, TurnEvent.TTS_URL) is TurnState.PLAYING


def test_terminal_events_finish_without_tts():
    for state in (TurnState.UPLINKING, TurnState.WAITING_FOR_TTS):
        for event in (TurnEvent.NO_SPEECH, TurnEvent.PIPELINE_ERROR,
                      TurnEvent.RUN_END):
            assert transition(state, event) is TurnState.FINISHED


def test_cancellation_is_explicit_and_cleanup_finishes_it():
    assert transition(TurnState.UPLINKING, TurnEvent.CANCEL) is TurnState.CANCELLED
    assert transition(TurnState.PLAYING, TurnEvent.CANCEL) is TurnState.CANCELLED
    assert transition(TurnState.CANCELLED, TurnEvent.TTS_COMPLETE) is TurnState.FINISHED


def test_finished_turn_rejects_new_events():
    try:
        transition(TurnState.FINISHED, TurnEvent.TTS_URL)
    except InvalidTransition:
        pass
    else:
        raise AssertionError("finished turn accepted a new event")
