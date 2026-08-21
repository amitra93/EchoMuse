from __future__ import annotations

import asyncio

import pytest
from aiohttp import ClientSession, WSMsgType
from aiohttp.test_utils import TestServer

from em_sendspin import SendspinListener


class FakePlayer:
    def __init__(self, name="Study", *, error=None, wait=False):
        self.name = name
        self.error = error
        self.wait = wait
        self.calls = []
        self.detached = False

    async def serve_websocket(self, websocket, *, expected_server_id=None):
        self.calls.append((websocket, expected_server_id))
        if self.error is not None:
            raise self.error
        if self.wait:
            async for _message in websocket:
                pass
        else:
            await websocket.close()

    def detach(self):
        self.detached = True


class FakeZeroconf:
    def __init__(self):
        self.registered = []
        self.unregistered = []
        self.closed = False

    async def async_register_service(self, service, **kwargs):
        self.registered.append((service, kwargs))

    async def async_unregister_service(self, service):
        self.unregistered.append(service)

    async def async_close(self):
        self.closed = True


class FakeServiceInfo:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


async def _connect(server, path):
    session = ClientSession()
    websocket = await session.ws_connect(server.make_url(path))
    return session, websocket


def test_unknown_client_is_rejected():
    async def run():
        listener = SendspinListener(advertise=False)
        server = TestServer(listener.app)
        await server.start_server()
        try:
            session, websocket = await _connect(server, "/sendspin/unknown")
            try:
                message = await websocket.receive()
                assert message.type in (WSMsgType.CLOSE, WSMsgType.CLOSED)
                assert websocket.close_code == 1008
            finally:
                await websocket.close()
                await session.close()
        finally:
            await server.close()

    asyncio.run(run())


def test_listener_rejects_invalid_port_and_double_start():
    with pytest.raises(ValueError, match="between 0 and 65535"):
        SendspinListener(port=65536)

    async def run():
        listener = SendspinListener(port=0, advertise=False)
        await listener.start()
        try:
            with pytest.raises(RuntimeError, match="already started"):
                await listener.start()
        finally:
            await listener.stop()

    asyncio.run(run())


def test_unpaired_client_is_rejected_by_adapter_policy():
    async def run():
        player = FakePlayer(error=PermissionError("pairing required"))
        listener = SendspinListener(advertise=False)
        await listener.register_player("study-id", player)
        server = TestServer(listener.app)
        await server.start_server()
        try:
            session, websocket = await _connect(server, "/sendspin/study-id")
            try:
                message = await websocket.receive()
                assert message.type in (WSMsgType.CLOSE, WSMsgType.CLOSED)
                assert websocket.close_code == 1008
                assert len(player.calls) == 1
            finally:
                await websocket.close()
                await session.close()
        finally:
            await server.close()

    asyncio.run(run())


def test_duplicate_connections_route_to_one_registered_player():
    async def run():
        player = FakePlayer()
        listener = SendspinListener(advertise=False)
        await listener.register_player("study-id", player)
        with pytest.raises(ValueError, match="already registered"):
            await listener.register_player("study-id", FakePlayer("replacement"))

        server = TestServer(listener.app)
        await server.start_server()
        try:
            first_session, first = await _connect(server, "/sendspin/study-id")
            second_session, second = await _connect(server, "/sendspin/study-id")
            await first.receive()
            await second.receive()
            assert len(player.calls) == 2
            await first_session.close()
            await second_session.close()
        finally:
            await server.close()

    asyncio.run(run())


def test_connection_cleanup_and_unregister_detach_player():
    async def run():
        player = FakePlayer(wait=True)
        listener = SendspinListener(advertise=False)
        await listener.register_player("study-id", player)
        server = TestServer(listener.app)
        await server.start_server()
        session, websocket = await _connect(server, "/sendspin/study-id")
        assert len(listener._connections) == 1

        await websocket.close()
        await asyncio.sleep(0)
        assert len(listener._connections) == 0
        await session.close()
        await listener.unregister_player("study-id")
        assert listener.players == {}
        assert player.detached is True
        await server.close()

    asyncio.run(run())


def test_stop_closes_active_connections_and_cleans_registry():
    async def run():
        player = FakePlayer(wait=True)
        listener = SendspinListener(advertise=False)
        await listener.register_player("study-id", player)
        server = TestServer(listener.app)
        await server.start_server()
        session, websocket = await _connect(server, "/sendspin/study-id")
        assert len(listener._connections) == 1

        await listener.stop()

        assert len(listener._connections) == 0
        await websocket.receive()
        assert websocket.closed is True
        await session.close()
        await server.close()

    asyncio.run(run())


def test_mdns_records_follow_listener_lifecycle():
    async def run():
        zeroconf = FakeZeroconf()
        listener = SendspinListener(
            host="127.0.0.1",
            port=0,
            advertise_address="192.0.2.10",
            zeroconf_factory=lambda: zeroconf,
            service_info_factory=FakeServiceInfo,
        )
        player = FakePlayer("Study")
        await listener.register_player("study-id", player)
        await listener.start()
        assert len(zeroconf.registered) == 1
        service, kwargs = zeroconf.registered[0]
        assert service.type_ == "_sendspin._tcp.local."
        assert service.port != 0
        assert service.parsed_addresses == ["192.0.2.10"]
        assert service.properties == {
            "path": "/sendspin/study-id",
            "name": "Study",
        }
        assert kwargs == {"allow_name_change": True}

        await listener.unregister_player("study-id")
        assert zeroconf.unregistered == [service]
        assert player.detached is True
        await listener.stop()
        assert zeroconf.closed is True

    asyncio.run(run())


def test_register_after_start_advertises_immediately():
    async def run():
        zeroconf = FakeZeroconf()
        listener = SendspinListener(
            port=0,
            advertise_address="192.0.2.10",
            zeroconf_factory=lambda: zeroconf,
            service_info_factory=FakeServiceInfo,
        )
        await listener.start()
        try:
            await listener.register_player("study-id", FakePlayer("Study"))
            assert len(zeroconf.registered) == 1
            service, _ = zeroconf.registered[0]
            assert service.port == listener.port
        finally:
            await listener.stop()

        assert len(zeroconf.unregistered) == 1
        assert zeroconf.closed is True

    asyncio.run(run())


def test_mdns_backend_is_closed_when_listener_has_no_players():
    async def run():
        zeroconf = FakeZeroconf()
        listener = SendspinListener(
            port=0,
            zeroconf_factory=lambda: zeroconf,
            service_info_factory=FakeServiceInfo,
        )
        await listener.start()
        assert zeroconf.registered == []
        await listener.stop()
        assert zeroconf.closed is True

    asyncio.run(run())
