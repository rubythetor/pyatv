"""Companion application liveness and internal power observations."""

import asyncio

import pytest
import pytest_asyncio

from pyatv import exceptions
from pyatv.const import PowerState, Protocol
from pyatv.core import UpdatedState, create_core
from pyatv.protocols.companion import CompanionPower
from pyatv.protocols.companion.api import CompanionAPI, SystemStatus
from pyatv.support.state_producer import StateProducer

from tests.fake_device.companion import CompanionServiceFlags, FakeCompanionService
from tests.utils import until


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
async def test_internal_power_observations(api, companion_state):
    client, _ = api
    observed = []
    client.core.state_dispatcher.listen_to(UpdatedState.Power, observed.append)
    power = CompanionPower(client)
    await power.initialize()
    await until(lambda: bool(observed))
    assert observed[-1].value == PowerState.On
    companion_state.system_status = SystemStatus.Asleep
    await until(lambda: observed[-1].value == PowerState.Off)
    assert observed[-1].protocol == Protocol.Companion


@pytest.mark.asyncio
@pytest.mark.use_heartbeat
async def test_heartbeat_reports_unresponsive_peer(api, companion_state, monkeypatch):
    client, observer = api
    monkeypatch.setattr("pyatv.protocols.companion.api.HEARTBEAT_TIMEOUT", 0.01)
    protocol = client._protocol
    requests = []
    for peer in companion_state.clients:
        monkeypatch.setattr(peer, "handle_fetchattentionstate", requests.append)
    await until(lambda: bool(observer.errors))
    assert len(observer.errors) == 1
    assert isinstance(observer.errors[0], asyncio.TimeoutError)
    assert len(requests) == 2
    assert not protocol._queues


@pytest.mark.asyncio
@pytest.mark.use_heartbeat
async def test_unsupported_heartbeat_reply_keeps_connection(api, companion_state):
    client, observer = api
    companion_state.set_flag_state(CompanionServiceFlags.SYSTEM_STATUS_SUPPORTED, False)
    # Unlike the liveness request, an ordinary power query must still raise.
    with pytest.raises(exceptions.ProtocolError):
        await client.fetch_attention_state()
    await client._heartbeat(None)
    assert not observer.errors
    assert (await client.app_list())["_c"] == {}


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
@pytest.mark.use_heartbeat
async def test_disconnect_cancels_heartbeat(api, companion_state, monkeypatch):
    client, observer = api
    received = asyncio.Event()
    for peer in companion_state.clients:
        monkeypatch.setattr(
            peer, "handle_fetchattentionstate", lambda request: received.set()
        )
    await received.wait()
    await client.disconnect()
    assert client._heartbeat_task is None
    assert not observer.errors


@pytest.mark.asyncio
@pytest.mark.use_heartbeat
async def test_standby_recovers_unresponsive_session(api, companion_state, monkeypatch):
    client, observer = api
    monkeypatch.setattr("pyatv.protocols.companion.api.HEARTBEAT_TIMEOUT", 0.01)
    power = CompanionPower(client)
    await power.initialize()
    companion_state.system_status = SystemStatus.Asleep
    await until(lambda: power.power_state == PowerState.Off)
    previous = client._protocol
    old_connection = client._connection
    companion_state.interests.clear()
    for peer in list(companion_state.clients):
        monkeypatch.setattr(peer, "handle_fetchattentionstate", lambda request: None)
    await until(lambda: client._protocol is not previous or observer.errors)
    assert not observer.errors
    await until(lambda: client._recovery_task is None)
    assert (await client.app_list())["_c"] == {}
    assert {"_iMC", "SystemStatus", "TVSystemStatus"} <= companion_state.interests
    # EOF queued by the retired transport cannot close its replacement.
    old_connection.connection_lost(OSError("retired connection"))
    assert not observer.errors
    companion_state.system_status = SystemStatus.Awake
    await until(lambda: power.power_state == PowerState.On)


@pytest.mark.asyncio
async def test_standby_recovery_failure_is_reported_once(
    api, companion_state, monkeypatch
):
    client, observer = api
    power = CompanionPower(client)
    await power.initialize()
    companion_state.system_status = SystemStatus.Asleep
    await until(lambda: power.power_state == PowerState.Off)
    monkeypatch.setattr("pyatv.protocols.companion.api.HEARTBEAT_TIMEOUT", 0.01)
    monkeypatch.setattr(
        FakeCompanionService, "handle_fetchattentionstate", lambda *args: None
    )
    connection = client._connection
    connection.close()
    await until(lambda: bool(observer.errors))
    assert len(observer.errors) == 1
    assert isinstance(observer.errors[0], asyncio.TimeoutError)
    assert client._protocol is None
    connection.connection_lost(OSError("late EOF"))
    assert len(observer.errors) == 1
    with pytest.raises(exceptions.InvalidStateError):
        await client.app_list()


@pytest.mark.asyncio
async def test_disconnect_during_recovery_cannot_reopen(
    api, companion_state, monkeypatch
):
    client, observer = api
    power = CompanionPower(client)
    await power.initialize()
    companion_state.system_status = SystemStatus.Asleep
    await until(lambda: power.power_state == PowerState.Off)
    request_received = asyncio.Event()
    monkeypatch.setattr(
        FakeCompanionService, "handle__systeminfo", lambda *args: request_received.set()
    )
    client._connection.close()
    await asyncio.wait_for(request_received.wait(), 0.5)
    await client.disconnect()
    assert not observer.errors
    assert client._protocol is None
    assert client._recovery_task is None
    with pytest.raises(exceptions.InvalidStateError):
        await client.connect()


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


@pytest.mark.asyncio
async def test_recovery_does_not_replay_inflight_command(
    api, companion_state, monkeypatch
):
    client, observer = api
    power = CompanionPower(client)
    await power.initialize()
    companion_state.system_status = SystemStatus.Asleep
    await until(lambda: power.power_state == PowerState.Off)
    commands = []
    for peer in companion_state.clients:
        monkeypatch.setattr(
            peer, "handle_fetchlaunchableapplicationsevent", commands.append
        )
    request = asyncio.create_task(client.app_list())
    await until(lambda: bool(commands))
    client._connection.close()
    with pytest.raises(exceptions.ProtocolError) as error:
        await asyncio.wait_for(request, 0.5)
    assert isinstance(error.value.__cause__, exceptions.ConnectionLostError)
    await until(lambda: client._recovery_task is None)
    assert not observer.errors
    assert len(commands) == 1
    assert (await client.app_list())["_c"] == {}


@pytest.mark.asyncio
async def test_recovery_has_total_deadline(api, companion_state, monkeypatch):
    client, observer = api
    power = CompanionPower(client)
    await power.initialize()
    companion_state.system_status = SystemStatus.Asleep
    await until(lambda: power.power_state == PowerState.Off)
    monkeypatch.setattr("pyatv.protocols.companion.api.RECOVERY_TIMEOUT", 0.02)
    monkeypatch.setattr(FakeCompanionService, "handle__systeminfo", lambda *args: None)
    client._connection.close()
    await until(lambda: bool(observer.errors), timeout=0.5)
    assert len(observer.errors) == 1
    assert isinstance(observer.errors[0], asyncio.TimeoutError)
    assert client._protocol is None


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [None, {}, {"state": None}, {"state": "invalid"}])
async def test_malformed_heartbeat_does_not_change_power(
    api, companion_state, monkeypatch, content
):
    client, observer = api
    power = CompanionPower(client)
    await power.initialize()
    for peer in companion_state.clients:
        monkeypatch.setattr(
            peer,
            "handle_fetchattentionstate",
            lambda request: peer.send_response(request, content),
        )
    await client._heartbeat(None)
    assert power.power_state == PowerState.On
    assert not observer.errors
    assert (await client.app_list())["_c"] == {}
