"""Sendspin group-control proxy for synchronized EchoMuse music players."""

from __future__ import annotations

from homeassistant.components.media_player import (
    MediaPlayerEntity,
    MediaPlayerEntityFeature,
    MediaPlayerState,
)

from .client import ControllerError
from .const import CAP_MUSIC_SYNC, DOMAIN
from .entities import EchoCoordinatorEntity, add_dynamic_entities

_RAW_FLOOR = 47   # device volumeButtonFloor (level 1)
_RAW_MAX = 127    # device volumeMax (level 11)


class EchoSendspinMediaPlayer(EchoCoordinatorEntity, MediaPlayerEntity):
    _attr_name = "Music"
    _attr_media_content_type = "music"
    _attr_supported_features = (
        MediaPlayerEntityFeature.PLAY_MEDIA
        | MediaPlayerEntityFeature.PAUSE
        | MediaPlayerEntityFeature.STOP
        | MediaPlayerEntityFeature.NEXT_TRACK
        | MediaPlayerEntityFeature.PREVIOUS_TRACK
        | MediaPlayerEntityFeature.SEEK
        | MediaPlayerEntityFeature.VOLUME_SET
        | MediaPlayerEntityFeature.VOLUME_MUTE
    )

    def __init__(self, coordinator, client, device_id: str):
        super().__init__(coordinator, device_id)
        self.client = client
        self._attr_unique_id = f"{device_id}_sendspin_music"

    @property
    def state(self) -> MediaPlayerState:
        state = self.record.get("sendspin_state")
        value = state.get("state") if isinstance(state, dict) else (self.record.get("media_state") or "idle")
        try:
            return MediaPlayerState(value)
        except ValueError:
            return MediaPlayerState.IDLE

    @property
    def volume_level(self) -> float | None:
        """Master volume level mapped to the Echo device's 1-11 range (0.01-1.00), 0.0 when muted."""
        if self.is_volume_muted:
            return 0.0
        value = self.record.get("volume")
        if not isinstance(value, (int, float)):
            return None
        raw = round(value * _RAW_MAX)
        if raw <= 0:
            return 0.0
        if raw <= _RAW_FLOOR:
            return 0.01
        if raw >= _RAW_MAX:
            return 1.0
        return round(0.01 + ((raw - _RAW_FLOOR) / (_RAW_MAX - _RAW_FLOOR)) * 0.99, 4)

    @property
    def is_volume_muted(self) -> bool:
        return bool(self.record.get("muted", False))

    @property
    def media_title(self) -> str | None:
        state = self.record.get("sendspin_state")
        if isinstance(state, dict) and state.get("title"):
            return state["title"]
        return self.record.get("sendspin_title") or None

    @property
    def media_artist(self) -> str | None:
        state = self.record.get("sendspin_state")
        if isinstance(state, dict) and state.get("artist"):
            return state["artist"]
        return self.record.get("sendspin_artist") or None

    async def _sendspin_command(self, command: str, **extra) -> None:
        try:
            await self.client.async_media_command(
                self.device_id,
                {"sendspin": True, "command": command, **extra},
            )
        except ControllerError as exc:
            raise RuntimeError(str(exc)) from exc

    async def async_media_play(self) -> None:
        await self._sendspin_command("play")

    async def async_media_pause(self) -> None:
        await self._sendspin_command("pause")

    async def async_media_stop(self) -> None:
        await self._sendspin_command("stop")

    async def async_media_next_track(self) -> None:
        await self._sendspin_command("next")

    async def async_media_previous_track(self) -> None:
        await self._sendspin_command("previous")

    async def async_media_seek(self, position: float) -> None:
        await self._sendspin_command("seek", position_ms=int(position * 1000))

    async def async_set_volume_level(self, volume: float) -> None:
        """Set volume mapped to device raw level (0.01->47 / Level 1, 1.00->127 / Level 11, 0.0->Mute)."""
        if volume <= 0.0:
            if not self.is_volume_muted:
                try:
                    await self.client.async_media_command(
                        self.device_id,
                        {"command": "mute_toggle"},
                    )
                except ControllerError as exc:
                    raise RuntimeError(str(exc)) from exc
            return

        if self.is_volume_muted:
            try:
                await self.client.async_media_command(
                    self.device_id,
                    {"command": "mute_toggle"},
                )
            except ControllerError as exc:
                raise RuntimeError(str(exc)) from exc

        if volume <= 0.01:
            raw = _RAW_FLOOR
        elif volume >= 1.0:
            raw = _RAW_MAX
        else:
            raw = round(_RAW_FLOOR + ((volume - 0.01) / 0.99) * (_RAW_MAX - _RAW_FLOOR))

        try:
            await self.client.async_media_command(
                self.device_id,
                {"volume": raw / _RAW_MAX},
            )
        except ControllerError as exc:
            raise RuntimeError(str(exc)) from exc

    async def async_mute_volume(self, mute: bool) -> None:
        if bool(mute) != bool(self.is_volume_muted):
            try:
                await self.client.async_media_command(
                    self.device_id,
                    {"command": "mute_toggle"},
                )
            except ControllerError as exc:
                raise RuntimeError(str(exc)) from exc

    async def async_play_media(self, media_type: str, media_id: str, **kwargs) -> None:
        try:
            await self.client.async_media_command(
                self.device_id,
                {"media_url": media_id},
            )
        except ControllerError as exc:
            raise RuntimeError(str(exc)) from exc


async def async_setup_entry(hass, entry, async_add_entities):
    data = hass.data[DOMAIN][entry.entry_id]
    coordinator = data["coordinator"]
    client = data["client"]
    entry.async_on_unload(add_dynamic_entities(
        coordinator,
        async_add_entities,
        lambda record: [EchoSendspinMediaPlayer(coordinator, client, record["device_id"])],
        capability=CAP_MUSIC_SYNC,
    ))
