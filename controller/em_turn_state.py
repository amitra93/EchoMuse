"""Pure state transitions for a streaming ESPHome voice turn."""

from __future__ import annotations

from enum import Enum


class TurnState(str, Enum):
    UPLINKING = "uplinking"
    WAITING_FOR_TTS = "waiting_for_tts"
    PLAYING = "playing"
    CANCELLED = "cancelled"
    FINISHED = "finished"


class TurnEvent(str, Enum):
    HA_VAD_END = "ha_vad_end"
    TTS_URL = "tts_url"
    NO_SPEECH = "no_speech"
    PIPELINE_ERROR = "pipeline_error"
    RUN_END = "run_end"
    CANCEL = "cancel"
    TTS_COMPLETE = "tts_complete"
    PLAYBACK_DRAINED = "playback_drained"


class InvalidTransition(ValueError):
    """Raised when an event cannot be applied to the current turn state."""


def barge_playback_fired(score: float, threshold: float, previous_score: float) -> bool:
    """Require two consecutive threshold crossings over active playback."""
    return score >= threshold and previous_score >= threshold


def transition(state: TurnState, event: TurnEvent) -> TurnState:
    """Apply one event to the turn state.

    RUN_END is deliberately ignored before HA has reached INTENT_END by the
    caller. Once represented as an event here, it is a terminal no-TTS result.
    """
    if state is TurnState.UPLINKING:
        if event in (TurnEvent.HA_VAD_END, TurnEvent.TTS_URL):
            return TurnState.WAITING_FOR_TTS
        if event in (TurnEvent.NO_SPEECH, TurnEvent.PIPELINE_ERROR,
                     TurnEvent.RUN_END):
            return TurnState.FINISHED
        if event is TurnEvent.CANCEL:
            return TurnState.CANCELLED

    elif state is TurnState.WAITING_FOR_TTS:
        if event is TurnEvent.TTS_URL:
            return TurnState.PLAYING
        if event in (TurnEvent.NO_SPEECH, TurnEvent.PIPELINE_ERROR,
                     TurnEvent.RUN_END, TurnEvent.PLAYBACK_DRAINED):
            return TurnState.FINISHED
        if event is TurnEvent.CANCEL:
            return TurnState.CANCELLED

    elif state is TurnState.PLAYING:
        if event in (TurnEvent.TTS_COMPLETE, TurnEvent.PLAYBACK_DRAINED):
            return TurnState.FINISHED
        if event is TurnEvent.CANCEL:
            return TurnState.CANCELLED

    elif state is TurnState.CANCELLED:
        if event in (TurnEvent.TTS_COMPLETE, TurnEvent.PLAYBACK_DRAINED,
                     TurnEvent.CANCEL):
            return TurnState.FINISHED

    elif state is TurnState.FINISHED:
        raise InvalidTransition(f"{event.value} after turn finished")

    raise InvalidTransition(f"{event.value} from {state.value}")
