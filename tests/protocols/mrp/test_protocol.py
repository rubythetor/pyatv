"""Unittests for pyatv.protocols.mrp.protocol."""

import asyncio
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from pyatv import exceptions
from pyatv.auth.hap_srp import SRPAuthHandler
from pyatv.conf import ManualService
from pyatv.const import Protocol
from pyatv.protocols.mrp import messages
from pyatv.protocols.mrp.connection import MrpConnection
from pyatv.protocols.mrp.protocol import (
    HEARTBEAT_INTERVAL,
    HEARTBEAT_RETRIES,
    MrpProtocol,
    heartbeat_loop,
)
from pyatv.settings import InfoSettings

from tests.fake_device import FakeAppleTV
from tests.utils import total_sleep_time, until


@pytest_asyncio.fixture
async def mrp_atv():
    atv = FakeAppleTV(asyncio.get_running_loop())
    atv.add_service(Protocol.MRP)
    await atv.start()
    yield atv
    await atv.stop()


@pytest_asyncio.fixture
async def mrp_protocol(mrp_atv):
    port = mrp_atv.get_port(Protocol.MRP)
    service = ManualService("mrp_id", Protocol.MRP, port, {})
    connection = MrpConnection("127.0.0.1", port, asyncio.get_running_loop())
    protocol = MrpProtocol(connection, SRPAuthHandler(), service, InfoSettings())
    yield protocol
    protocol.stop()


@pytest.mark.asyncio
@pytest.mark.use_heartbeat
async def test_heartbeat_loop(mrp_atv, mrp_protocol):
    await mrp_protocol.start()
    mrp_protocol.enable_heartbeat()

    mrp_state = mrp_atv.get_state(Protocol.MRP)
    await until(lambda: mrp_state.heartbeat_count >= 3)


@pytest.mark.asyncio
async def test_heartbeat_fail_closes_connection(stub_sleep):
    protocol = MagicMock()
    protocol.send_and_receive.side_effect = Exception()

    await heartbeat_loop(protocol)
    assert protocol.send_and_receive.call_count == 1 + HEARTBEAT_RETRIES
    assert total_sleep_time() == HEARTBEAT_INTERVAL

    protocol.connection.close.assert_called_once()


@pytest.mark.asyncio
async def test_stop_fails_pending_requests(mrp_protocol):
    pending = asyncio.create_task(mrp_protocol._receive("pending", 60))
    await asyncio.sleep(0)
    mrp_protocol.stop()
    with pytest.raises(exceptions.ConnectionLostError):
        await asyncio.wait_for(pending, 0.1)


@pytest.mark.asyncio
async def test_cancelled_request_is_removed(mrp_protocol):
    pending = asyncio.create_task(mrp_protocol._receive("pending", 60))
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert not mrp_protocol._outstanding


@pytest.mark.asyncio
async def test_resume_repeats_handshake(mrp_atv, mrp_protocol, monkeypatch):
    await mrp_protocol.start()
    # AirPlay detaches its old channel listener before closing. The plain MRP
    # transport used by this fixture normally treats every EOF as terminal.
    monkeypatch.setattr(mrp_protocol.connection, "connection_lost", lambda exc: None)
    mrp_protocol.suspend()
    with pytest.raises(exceptions.InvalidStateError):
        await mrp_protocol.send(messages.get_keyboard_session())
    await mrp_protocol.start()
    response = await mrp_protocol.send_and_receive(messages.get_keyboard_session())
    assert response.type == 23


@pytest.mark.asyncio
async def test_terminal_stop_cannot_resume(mrp_protocol):
    mrp_protocol.stop()
    with pytest.raises(exceptions.InvalidStateError):
        await mrp_protocol.start()
