import asyncio
import importlib
import types

import pytest

pytest.importorskip("homeassistant")

module = importlib.import_module("custom_components.echo_voice_satellite.select")
from custom_components.echo_voice_satellite.client import ControllerError  # noqa: E402


class _FakeCoordinator:
    def __init__(self, record):
        self.data = {"devices": [record]}
        self.control_available = True
        self.last_update_success = True


class _FakeEntry:
    def __init__(self, options=None):
        self.options = options or {}
        self.updates = []

    # Mirrors hass.config_entries.async_update_entry's signature enough
    # for this entity's call site.


class _FakeConfigEntries:
    def __init__(self, entry):
        self.entry = entry

    def async_update_entry(self, entry, options):
        entry.options = options


class _FakeHass:
    def __init__(self, entry):
        self.config_entries = _FakeConfigEntries(entry)


def _pipeline(name, pipeline_id):
    return types.SimpleNamespace(name=name, id=pipeline_id)


def _make(monkeypatch, pipelines, options=None, device_id="A"):
    monkeypatch.setattr(module.assist_pipeline, "async_get_pipelines", lambda hass: pipelines)
    entry = _FakeEntry(options=options)
    hass = _FakeHass(entry)
    coordinator = _FakeCoordinator({"device_id": device_id})
    entity = object.__new__(module.EchoAssistPipelineSelect)
    entity.coordinator = coordinator
    entity.hass = hass
    entity.entry = entry
    entity.device_id = device_id
    entity._observed = False
    entity._pipeline_ids = {}
    written = []
    entity.async_write_ha_state = lambda: written.append(True)
    return entity, entry, written


def test_options_lists_every_available_pipeline_by_name(monkeypatch):
    entity, _entry, _w = _make(monkeypatch, [_pipeline("Home", "id-1"), _pipeline("Kitchen", "id-2")])
    assert entity.options == ["Home", "Kitchen"]


def test_current_option_is_none_when_nothing_selected(monkeypatch):
    entity, _entry, _w = _make(monkeypatch, [_pipeline("Home", "id-1")])
    assert entity.current_option is None


def test_current_option_resolves_the_selected_pipeline_name(monkeypatch):
    entity, _entry, _w = _make(
        monkeypatch, [_pipeline("Home", "id-1"), _pipeline("Kitchen", "id-2")],
        options={"assist_pipeline_ids": {"A": "id-2"}},
    )
    assert entity.current_option == "Kitchen"


def test_current_option_is_none_for_a_stale_pipeline_id(monkeypatch):
    # The stored id no longer matches anything async_get_pipelines returns
    # (the pipeline was deleted in HA) — falls through to None rather than
    # raising or showing a stale name.
    entity, _entry, _w = _make(
        monkeypatch, [_pipeline("Home", "id-1")],
        options={"assist_pipeline_ids": {"A": "id-deleted"}},
    )
    assert entity.current_option is None


def test_select_option_persists_to_entry_options_scoped_by_device(monkeypatch):
    entity, entry, written = _make(
        monkeypatch, [_pipeline("Home", "id-1")],
        options={"assist_pipeline_ids": {"OTHER_DEVICE": "id-9"}},
    )
    entity.options  # populates _pipeline_ids — HA's frontend always reads
    # the option list before selecting one; async_select_option's
    # id lookup depends on it having run at least once already.

    import asyncio
    asyncio.run(entity.async_select_option("Home"))

    assert entry.options["assist_pipeline_ids"] == {"OTHER_DEVICE": "id-9", "A": "id-1"}
    assert written == [True]


def test_select_option_rejects_an_unknown_pipeline_name(monkeypatch):
    entity, _entry, _written = _make(monkeypatch, [_pipeline("Home", "id-1")])
    entity.options  # populate _pipeline_ids, mirroring what current_option/options does

    import asyncio
    with pytest.raises(ValueError, match="Unknown Assist pipeline"):
        asyncio.run(entity.async_select_option("Nonexistent"))


# ── EchoVolumeSelect ─────────────────────────────────────────────────────────
# 9 discrete steps replacing the earlier continuous slider (number.py,
# removed — __init__.py's _remove_stale_volume_number_entities). Fixture
# pattern matches test_button.py / the removed test_number.py.

class _FakeVolumeClient:
    def __init__(self):
        self.calls = []
        self.should_fail = False

    async def async_media_command(self, device_id, body):
        self.calls.append((device_id, body))
        if self.should_fail:
            raise ControllerError("media_command_failed")


def _make_volume(record, client=None):
    coordinator = _FakeCoordinator(record)
    client = client or _FakeVolumeClient()
    entity = object.__new__(module.EchoVolumeSelect)
    entity.coordinator = coordinator
    entity.client = client
    entity.device_id = record["device_id"]
    entity._observed = False
    return entity, client


def test_volume_options_are_the_nine_levels():
    entity, _client = _make_volume({"device_id": "A"})
    assert entity.options == [str(i) for i in range(1, 10)]


def test_volume_current_option_is_none_when_never_reported():
    entity, _client = _make_volume({"device_id": "A"})
    assert entity.current_option is None


def test_volume_current_option_maps_the_button_floor_to_level_one():
    # 47/127 is device volumeButtonFloor exactly — where a physical press
    # already stops, and this entity's own level 1.
    entity, _client = _make_volume({"device_id": "A", "volume": 47 / 127})
    assert entity.current_option == "1"


def test_volume_current_option_maps_unity_gain_to_level_nine():
    entity, _client = _make_volume({"device_id": "A", "volume": 1.0})
    assert entity.current_option == "9"


def test_volume_current_option_snaps_an_off_table_value_to_the_nearest_level():
    # A value that predates this entity (e.g. set by the old slider) won't
    # land exactly on one of the 9 raw levels — must still resolve to SOME
    # option rather than None.
    entity, _client = _make_volume({"device_id": "A", "volume": 0.10})  # raw ~13
    assert entity.current_option == "1"  # nearest to raw 47


def test_select_option_sends_the_raw_level_for_level_five():
    # Level 5 is the midpoint: raw 47 + 4*10 = 87.
    entity, client = _make_volume({"device_id": "A"})
    asyncio.run(entity.async_select_option("5"))
    assert client.calls == [("A", {"volume": 87 / 127})]


def test_select_option_level_one_reaches_the_button_floor_not_below_it():
    entity, client = _make_volume({"device_id": "A"})
    asyncio.run(entity.async_select_option("1"))
    assert client.calls == [("A", {"volume": 47 / 127})]


def test_select_option_level_nine_reaches_full_scale():
    entity, client = _make_volume({"device_id": "A"})
    asyncio.run(entity.async_select_option("9"))
    assert client.calls == [("A", {"volume": 1.0})]


def test_select_option_rejects_an_out_of_range_level():
    entity, _client = _make_volume({"device_id": "A"})
    with pytest.raises(ValueError, match="Unknown volume level"):
        asyncio.run(entity.async_select_option("10"))


def test_select_option_rejects_level_zero_rather_than_wrapping_to_level_nine():
    # A bare list index (self._LEVELS[level - 1]) with no bounds check would
    # silently accept "0" as Python negative-index -1 — the LAST entry,
    # i.e. full volume, which is the opposite of what "0" would mean and
    # with no error to notice it by. Caught in review before ever shipping.
    entity, client = _make_volume({"device_id": "A"})
    with pytest.raises(ValueError, match="Unknown volume level"):
        asyncio.run(entity.async_select_option("0"))
    assert client.calls == []


def test_select_option_rejects_a_non_numeric_level():
    entity, _client = _make_volume({"device_id": "A"})
    with pytest.raises(ValueError, match="Unknown volume level"):
        asyncio.run(entity.async_select_option("not-a-level"))


def test_select_option_converts_controller_error_to_value_error():
    client = _FakeVolumeClient()
    client.should_fail = True
    entity, client = _make_volume({"device_id": "A"}, client=client)
    with pytest.raises(ValueError):
        asyncio.run(entity.async_select_option("5"))
