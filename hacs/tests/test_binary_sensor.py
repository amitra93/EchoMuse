import importlib

import pytest

pytest.importorskip("homeassistant")

module = importlib.import_module("custom_components.echo_voice_satellite.binary_sensor")


class _FakeCoordinator:
    def __init__(self, record):
        self.data = {"devices": [record]}
        self.control_available = True
        self.last_update_success = True


def _make(cls, record):
    coordinator = _FakeCoordinator(record)
    entity = object.__new__(cls)
    entity.coordinator = coordinator
    entity.device_id = record["device_id"]
    entity._observed = False
    return entity


@pytest.mark.parametrize("connected", [True, False])
def test_online_sensor_mirrors_connected_flag(connected):
    entity = _make(module.EchoOnlineSensor, {"device_id": "A", "connected": connected})
    assert entity.is_on is connected


def test_binary_sensor_module_no_longer_defines_a_mute_entity():
    # Privacy mute moved to switch.py's EchoMuteSwitch once mute_toggle
    # existed on the wire — a switch replaces this read-only sensor AND
    # the momentary button.py button with one entity.
    assert not hasattr(module, "EchoMutedSensor")
