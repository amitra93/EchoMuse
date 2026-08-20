"""Select the Assist pipeline used by one satellite, and (EchoVolumeSelect)
master volume as 9 discrete steps.
"""

from __future__ import annotations

from homeassistant.components import assist_pipeline
from homeassistant.components.select import SelectEntity

from .client import ControllerError
from .const import DOMAIN
from .entities import EchoCoordinatorEntity, add_dynamic_entities


async def async_setup_entry(hass, entry, async_add_entities):
    data = hass.data[DOMAIN][entry.entry_id]
    coordinator = data["coordinator"]
    client = data["client"]
    entry.async_on_unload(add_dynamic_entities(
        coordinator, async_add_entities,
        lambda record: [
            EchoAssistPipelineSelect(hass, coordinator, entry, record["device_id"]),
            EchoVolumeSelect(coordinator, client, record["device_id"]),
        ],
    ))


class EchoAssistPipelineSelect(EchoCoordinatorEntity, SelectEntity):
    _attr_name = "Assist pipeline"

    def __init__(self, hass, coordinator, entry, device_id: str):
        super().__init__(coordinator, device_id)
        self.hass = hass
        self.entry = entry
        self._attr_unique_id = f"{device_id}_assist_pipeline"
        self._pipeline_ids: dict[str, str] = {}

    @property
    def options(self) -> list[str]:
        pipelines = assist_pipeline.async_get_pipelines(self.hass)
        self._pipeline_ids = {pipeline.name: pipeline.id for pipeline in pipelines}
        return list(self._pipeline_ids)

    @property
    def current_option(self) -> str | None:
        self.options
        selected_id = self.entry.options.get("assist_pipeline_ids", {}).get(self.device_id)
        if not selected_id:
            return None
        for name, pipeline_id in self._pipeline_ids.items():
            if pipeline_id == selected_id:
                return name
        return None

    async def async_select_option(self, option: str) -> None:
        pipeline_id = self._pipeline_ids.get(option)
        if pipeline_id is None:
            raise ValueError(f"Unknown Assist pipeline: {option}")
        selected = {
            **self.entry.options.get("assist_pipeline_ids", {}),
            self.device_id: pipeline_id,
        }
        self.hass.config_entries.async_update_entry(
            self.entry, options={**self.entry.options, "assist_pipeline_ids": selected},
        )
        self.async_write_ha_state()


# Module-level, not class-body: a class-body list comprehension can only
# see the enclosing class namespace through its OUTERMOST for-clause's
# iterable, not through the comprehension's body — `_RAW_FLOOR`/`_RAW_MAX`
# referenced inside the body raised NameError at class-definition time
# (caught immediately by the real-HA test run, not by py_compile, which
# only checks syntax).
_VOLUME_RAW_FLOOR = 47   # device volumeButtonFloor
_VOLUME_RAW_MAX = 127    # device volumeMax / em_volume.DEVICE_VOLUME_MAX
_VOLUME_LEVELS = [
    round(_VOLUME_RAW_FLOOR + i * (_VOLUME_RAW_MAX - _VOLUME_RAW_FLOOR) / 8)
    for i in range(9)
]
_VOLUME_OPTIONS = [str(i) for i in range(1, 10)]


class EchoVolumeSelect(EchoCoordinatorEntity, SelectEntity):
    """Master volume as 9 discrete steps, replacing the earlier continuous
    slider (number.py, removed — see __init__.py's
    _remove_stale_volume_number_entities).

    The slider's own send path was verified end-to-end with no floor and no
    snap-back — direct calls to the same controller endpoint this entity
    uses landed anywhere from 0% to 100% instantly and held (checked over
    10s of polling, 2026-08-19) — so the "stuck around 37%" users saw was an
    HA-frontend slider-drag artefact, not a backend bug. A discrete select
    sends one full, deliberate request per press instead of a stream of
    intermediate drag values, which sidesteps that whole class of problem
    regardless of its exact cause.

    The 9 levels are the device's own physical-button range
    (device/internal/server/volume.go: `volumeButtonFloor`=47..`volumeMax`
    =127 on the raw 0-127 tinymix scale, matching em_volume.py's
    DEVICE_VOLUME_MAX), evenly divided into 8 steps — level 1 is exactly
    where a physical press already floors to (37%, never fully silent by
    button — the mute button is the way to actually go silent), level 9 is
    the codec's unity-gain ceiling (100%). Values are duplicated here rather
    than imported from em_volume.py: this package runs inside Home
    Assistant's own Python process with no access to the controller's
    modules.
    """

    _attr_name = "Volume"

    _RAW_FLOOR = _VOLUME_RAW_FLOOR
    _RAW_MAX = _VOLUME_RAW_MAX
    _LEVELS = _VOLUME_LEVELS
    _OPTIONS = _VOLUME_OPTIONS

    def __init__(self, coordinator, client, device_id: str):
        super().__init__(coordinator, device_id)
        self.client = client
        self._attr_unique_id = f"{device_id}_volume_level"

    @property
    def options(self) -> list[str]:
        return self._OPTIONS

    @property
    def current_option(self) -> str | None:
        value = self.record.get("volume")
        if not isinstance(value, (int, float)):
            return None
        raw = round(value * self._RAW_MAX)
        # Nearest level, not exact match — the device's actual raw level can
        # land off-table (e.g. a value set before this entity existed, or
        # any future controller-side rounding), and every raw value must
        # still map to SOME displayed level rather than showing nothing.
        nearest = min(range(len(self._LEVELS)), key=lambda i: abs(self._LEVELS[i] - raw))
        return self._OPTIONS[nearest]

    async def async_select_option(self, option: str) -> None:
        try:
            level = int(option)
        except ValueError:
            raise ValueError(f"Unknown volume level: {option}") from None
        # Explicit bounds check, not a bare list index: level=0 or a
        # negative level would otherwise silently wrap via Python's
        # negative-indexing rather than raise (e.g. level 0 -> index -1 ->
        # the LAST entry, level 9's raw value — the opposite of what was
        # asked for, and worse, no error at all).
        if not 1 <= level <= len(self._LEVELS):
            raise ValueError(f"Unknown volume level: {option}")
        raw = self._LEVELS[level - 1]
        try:
            await self.client.async_media_command(
                self.device_id, {"volume": raw / self._RAW_MAX})
        except ControllerError as exc:
            raise ValueError(str(exc)) from exc
