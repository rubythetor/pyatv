"""Companion request lifecycle tests."""

import asyncio

import pytest
import pytest_asyncio

from pyatv import exceptions
from pyatv.const import Protocol
from pyatv.core import create_core
from pyatv.protocols.companion.api import CompanionAPI
from pyatv.support.state_producer import StateProducer


class Observer:
    def __init__(self):
        self.errors = []

    def connection_lost(self, exc):
        self.errors.append(exc)


@pytest_asyncio.fixture
async def api(companion_conf, session_manager, core_dispatcher):
    listener = StateProducer()
    observer = Observer()
    listener.listener = observer
    core = await create_core(
        companion_conf,
        companion_conf.get_service(Protocol.Companion),
        device_listener=listener,
        session_manager=session_manager,
        core_dispatcher=core_dispatcher,
    )
    api = CompanionAPI(core)
    await api.connect()
    yield api, observer
    await api.disconnect()


@pytest.mark.asyncio
async def test_cancelled_companion_request_is_removed(
    api, companion_state, monkeypatch
):
    client, _ = api
    received = asyncio.Event()
    for peer in companion_state.clients:
        monkeypatch.setattr(
            peer, "handle_fetchattentionstate", lambda request: received.set()
        )
    request = asyncio.create_task(client.fetch_attention_state())
    await received.wait()
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    assert not client._protocol._queues


@pytest.mark.asyncio
async def test_stop_fails_pending_request_without_replay(
    api, companion_state, monkeypatch
):
    client, _ = api
    received = asyncio.Event()
    for peer in companion_state.clients:
        monkeypatch.setattr(
            peer, "handle_fetchattentionstate", lambda request: received.set()
        )
    request = asyncio.create_task(client.fetch_attention_state())
    await received.wait()
    client._protocol.stop()
    with pytest.raises(exceptions.ProtocolError) as error:
        await asyncio.wait_for(request, 0.1)
    assert isinstance(error.value.__cause__, exceptions.ConnectionLostError)
