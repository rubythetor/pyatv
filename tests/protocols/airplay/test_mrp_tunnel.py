"""AirPlay tunnel lifecycle tests with a fake MRP receiver."""

import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio

from pyatv import exceptions
from pyatv.auth.hap_pairing import parse_credentials
from pyatv.auth.server_auth import CLIENT_CREDENTIALS
from pyatv.conf import AppleTV, ManualService
from pyatv.const import DeviceState, PowerState, Protocol
from pyatv.core import UpdatedState, create_core
from pyatv.interface import Metadata, PushUpdater, RemoteControl
from pyatv.protocols import airplay
from pyatv.protocols.airplay.ap2_session import AP2Session
from pyatv.protocols.mrp import messages, protobuf
from pyatv.support.state_producer import StateProducer

from tests.fake_device.mrp import FakeMrpService, FakeMrpState, FakeMrpUseCases
from tests.utils import until


class Observer:
    def __init__(self):
        self.closed = 0
        self.errors = []
        self.playing = []

    def connection_closed(self):
        self.closed += 1

    def connection_lost(self, error):
        self.errors.append(error)

    def playstatus_update(self, updater, playing):
        self.playing.append(playing)

    def playstatus_error(self, updater, error):
        self.errors.append(error)


class FakeDataChannel(FakeMrpService):
    """Replace framing/encryption only; use the existing MRP receiver handlers."""

    def __init__(self, state):
        super().__init__(state, None, asyncio.get_running_loop())
        self.listener = None
        self.closed = False
        self.keyboard_requests = 0

    def send_protobuf(self, message):
        name = message.Type.Name(message.type).lower().replace("_message", "")
        if name == "get_keyboard_session":
            self.keyboard_requests += 1
        getattr(self, "handle_" + name)(message, protobuf.extract_inner(message))

    def send_to_client(self, message):
        if self.listener:
            self.loop.call_soon(self.listener.handle_protobuf, message)

    def close(self):
        if not self.closed:
            self.closed = True
            self.state.clients.remove(self)
            if self.listener:
                self.loop.call_soon(self.listener.handle_connection_lost, None)


@pytest_asyncio.fixture
async def tunnel(monkeypatch, session_manager, core_dispatcher):
    state = FakeMrpState()
    sessions = []
    feedback = asyncio.Queue()
    setup_gate = asyncio.Event()
    setup_gate.set()
    env = SimpleNamespace(fail_setup=False, fail_keyboard=False, early_state=False)

    class Session(AP2Session):
        async def connect(self):
            sessions.append(self)
            await setup_gate.wait()
            if env.fail_setup:
                raise OSError("receiver unreachable")
            self.rtsp = self

        async def setup_remote_control(self):
            channel = FakeDataChannel(state)
            if env.fail_keyboard:
                channel.handle_get_keyboard_session = lambda *args: None
            if env.early_state:
                original_info = channel.handle_device_info

                def early_info(message, inner):
                    for identifier in state.states:
                        state.update_state(identifier)
                    state.set_active_player(state.active_player)
                    original_info(message, inner)

                channel.handle_device_info = early_info
                channel.handle_client_updates_config = (
                    lambda message, inner: channel.send_to_client(
                        messages.create(0, identifier=message.identifier)
                    )
                )
            self.data_channel = channel
            self._channels.append(channel)

        async def feedback(self):
            result = await feedback.get()
            if isinstance(result, Exception):
                raise result

    monkeypatch.setattr(airplay, "AP2Session", Session)
    conf = AppleTV("127.0.0.1", "Test receiver")
    service = ManualService("test", Protocol.AirPlay, 7000, {})
    conf.add_service(service)
    observer = Observer()
    device_listener = StateProducer()
    device_listener.listener = observer
    core = await create_core(
        conf,
        service,
        session_manager=session_manager,
        device_listener=device_listener,
        core_dispatcher=core_dispatcher,
    )
    setup = airplay._create_mrp_tunnel_data(core, parse_credentials(CLIENT_CREDENTIALS))
    await setup.connect()
    push = setup.interfaces[PushUpdater]
    push.listener = observer
    push.start()
    env.core = core
    env.setup = setup
    env.sessions = sessions
    env.state = state
    env.usecase = FakeMrpUseCases(state)
    env.observer = observer
    env.feedback = feedback
    env.gate = setup_gate
    env.metadata = setup.interfaces[Metadata]
    env.remote = setup.interfaces[RemoteControl]
    env.push = push
    env.power = core.state_dispatcher.create_copy(Protocol.Companion)
    yield env
    await asyncio.gather(*setup.close())


async def power(tunnel, state):
    await asyncio.gather(*tunnel.power.dispatch(UpdatedState.Power, state))
    await asyncio.sleep(0)


def disconnect(channel, error=115):
    message = messages.create(protobuf.SET_CONNECTION_STATE_MESSAGE)
    protobuf.extract_inner(message).state = 3
    message.errorCode = error
    channel.send_to_client(message)


async def sleep(tunnel):
    await power(tunnel, PowerState.Off)
    channel = tunnel.sessions[-1].data_channel
    disconnect(channel)
    await until(lambda: channel.closed)
    return channel


async def wake(tunnel):
    previous = len(tunnel.sessions)
    await power(tunnel, PowerState.On)
    await until(lambda: len(tunnel.sessions) == previous + 1)
    await until(lambda: tunnel.sessions[-1].data_channel is not None)
    await until(lambda: tunnel.sessions[-1].data_channel.keyboard_requests == 1)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_sleep_retires_session_without_disconnect(tunnel):
    await sleep(tunnel)
    assert tunnel.push.active
    assert not tunnel.observer.closed
    assert not tunnel.observer.errors
    with pytest.raises(exceptions.InvalidStateError):
        await tunnel.remote.play()


@pytest.mark.asyncio
async def test_wake_restores_commands_and_push_updates(tunnel):
    tunnel.usecase.example_video(title="Before sleep")
    await until(
        lambda: any(item.title == "Before sleep" for item in tunnel.observer.playing)
    )
    await sleep(tunnel)
    tunnel.usecase.example_video(title="After wake")
    await wake(tunnel)
    await until(
        lambda: any(item.title == "After wake" for item in tunnel.observer.playing)
    )
    await tunnel.remote.play()
    assert tunnel.state.last_button_pressed == "play"
    assert tunnel.push.active
    assert not tunnel.observer.errors
    assert not tunnel.observer.closed


@pytest.mark.asyncio
async def test_wake_does_not_keep_stale_playback(tunnel):
    tunnel.usecase.example_video(title="Old film")
    await until(
        lambda: any(item.title == "Old film" for item in tunnel.observer.playing)
    )
    await sleep(tunnel)
    tunnel.state.states.clear()
    tunnel.state.active_player = None
    await wake(tunnel)
    playing = await tunnel.metadata.playing()
    assert playing.title is None
    assert playing.device_state == DeviceState.Idle


@pytest.mark.asyncio
async def test_retired_channel_cannot_close_replacement(tunnel):
    old = tunnel.sessions[-1].data_channel
    listener = old.listener
    await sleep(tunnel)
    await wake(tunnel)
    listener.handle_connection_lost(OSError("late EOF"))
    message = messages.create(protobuf.SET_CONNECTION_STATE_MESSAGE)
    message.errorCode = 104
    protobuf.extract_inner(message).state = 3
    listener.handle_protobuf(message)
    await tunnel.remote.play()
    assert not tunnel.observer.errors
    assert not tunnel.observer.closed


@pytest.mark.asyncio
async def test_duplicate_wake_creates_one_session(tunnel):
    await sleep(tunnel)
    tunnel.gate.clear()
    await power(tunnel, PowerState.On)
    await power(tunnel, PowerState.On)
    await until(lambda: len(tunnel.sessions) == 2)
    tunnel.gate.set()
    await until(lambda: tunnel.sessions[-1].data_channel is not None)
    assert len(tunnel.sessions) == 2


@pytest.mark.asyncio
async def test_wake_setup_failure_is_reported(tunnel):
    await sleep(tunnel)
    tunnel.fail_setup = True
    await power(tunnel, PowerState.On)
    await until(lambda: bool(tunnel.observer.errors))
    assert len(tunnel.observer.errors) == 1


@pytest.mark.asyncio
async def test_close_during_wake_does_not_reconnect(tunnel):
    await sleep(tunnel)
    tunnel.gate.clear()
    await power(tunnel, PowerState.On)
    await until(lambda: len(tunnel.sessions) == 2)
    await asyncio.gather(*tunnel.setup.close())
    tunnel.gate.set()
    await power(tunnel, PowerState.On)
    assert tunnel.sessions[-1].data_channel is None
    assert not tunnel.push.active
    assert not tunnel.observer.errors


@pytest.mark.asyncio
async def test_real_feedback_failure_is_reported(tunnel):
    await tunnel.feedback.put(TimeoutError())
    await tunnel.feedback.put(TimeoutError())
    await until(lambda: bool(tunnel.observer.errors))
    assert len(tunnel.observer.errors) == 1


@pytest.mark.asyncio
async def test_sleep_without_companion_keeps_failure_detection(tunnel):
    channel = tunnel.sessions[-1].data_channel
    disconnect(channel)
    await asyncio.sleep(0)
    assert not channel.closed
    await tunnel.feedback.put(TimeoutError())
    await tunnel.feedback.put(TimeoutError())
    await until(lambda: bool(tunnel.observer.errors))


@pytest.mark.asyncio
async def test_unannounced_mrp_disconnect_is_reported(tunnel):
    disconnect(tunnel.sessions[-1].data_channel, error=104)
    await until(lambda: bool(tunnel.observer.errors))


@pytest.mark.asyncio
async def test_sleep_announcement_before_companion_off(tunnel):
    channel = tunnel.sessions[-1].data_channel
    disconnect(channel)
    await asyncio.sleep(0)
    await power(tunnel, PowerState.Off)
    await until(lambda: channel.closed)
    assert not tunnel.observer.errors


@pytest.mark.asyncio
async def test_commands_rejected_until_wake_handshake_completes(tunnel):
    await sleep(tunnel)
    tunnel.fail_keyboard = True
    await power(tunnel, PowerState.On)
    await until(lambda: len(tunnel.sessions) == 2)
    await until(lambda: tunnel.sessions[-1].data_channel is not None)
    await until(lambda: tunnel.sessions[-1].data_channel.keyboard_requests == 1)
    with pytest.raises(exceptions.InvalidStateError):
        await tunnel.remote.play()


@pytest.mark.asyncio
async def test_wake_requires_mrp_response(tunnel, monkeypatch):
    monkeypatch.setattr("pyatv.protocols.airplay.mrp_tunnel.CONNECT_TIMEOUT", 0.05)
    await sleep(tunnel)
    tunnel.fail_keyboard = True
    await power(tunnel, PowerState.On)
    await until(lambda: bool(tunnel.observer.errors))
    assert isinstance(tunnel.observer.errors[0], asyncio.TimeoutError)
    assert len(tunnel.observer.errors) == 1


@pytest.mark.asyncio
async def test_two_sleep_wake_cycles_preserve_subscription(tunnel):
    for title in ["First", "Second"]:
        await sleep(tunnel)
        tunnel.usecase.example_video(title=title)
        await wake(tunnel)
        await until(lambda: tunnel.observer.playing[-1].title == title)
        assert tunnel.push.active
    assert len(tunnel.sessions) == 3
    assert not tunnel.observer.errors
    assert not tunnel.observer.closed


@pytest.mark.asyncio
async def test_initial_state_push_before_device_info_is_preserved(tunnel):
    await sleep(tunnel)
    tunnel.usecase.example_video(title="Pushed during handshake")
    tunnel.early_state = True
    await wake(tunnel)
    assert (await tunnel.metadata.playing()).title == "Pushed during handshake"


@pytest.mark.asyncio
async def test_artwork_cache_is_fresh_after_wake(tunnel):
    tunnel.usecase.example_video(
        artwork=b"first", artwork_identifier="same", artwork_mimetype="image/png"
    )
    await until(lambda: tunnel.metadata.artwork_id == "same")
    assert (await tunnel.metadata.artwork()).bytes == b"first"
    await sleep(tunnel)
    tunnel.usecase.example_video(
        artwork=b"second", artwork_identifier="same", artwork_mimetype="image/png"
    )
    await wake(tunnel)
    assert (await tunnel.metadata.artwork()).bytes == b"second"


@pytest.mark.asyncio
async def test_sleep_during_wake_can_resume_again(tunnel):
    await sleep(tunnel)
    tunnel.fail_keyboard = True
    await power(tunnel, PowerState.On)
    await until(lambda: len(tunnel.sessions) == 2)
    await until(lambda: tunnel.sessions[-1].data_channel is not None)
    await until(lambda: tunnel.sessions[-1].data_channel.keyboard_requests == 1)
    await sleep(tunnel)
    tunnel.fail_keyboard = False
    await wake(tunnel)
    await tunnel.remote.play()
    assert not tunnel.observer.errors


@pytest.mark.asyncio
async def test_initial_handshake_failure_closes_partial_session(tunnel, monkeypatch):
    monkeypatch.setattr("pyatv.protocols.airplay.mrp_tunnel.CONNECT_TIMEOUT", 0.05)
    tunnel.fail_keyboard = True
    setup = airplay._create_mrp_tunnel_data(
        tunnel.core, parse_credentials(CLIENT_CREDENTIALS)
    )
    try:
        with pytest.raises(asyncio.TimeoutError):
            await setup.connect()
        assert tunnel.sessions[-1].data_channel is None
    finally:
        await asyncio.gather(*setup.close(), return_exceptions=True)
