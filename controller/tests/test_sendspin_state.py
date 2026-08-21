import asyncio
import sys
import types

import em_ha_sidechannels


def test_sendspin_state_event_contains_media_state(monkeypatch):
    async def run():
        events = []

        async def push(event):
            events.append(event)

        monkeypatch.setitem(sys.modules, "em_api", types.SimpleNamespace(_push_event=push))
        em_ha_sidechannels.sendspin_state(
            "device", "playing", volume=40, muted=True,
            title="Song", artist="Artist",
        )
        await asyncio.sleep(0)
        assert events == [{
            "type": "sendspin_state",
            "device_id": "device",
            "state": {
                "state": "playing", "volume": 0.4, "muted": True,
                "title": "Song", "artist": "Artist",
            },
        }]

    asyncio.run(run())
