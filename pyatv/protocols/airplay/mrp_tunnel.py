"""Lifecycle of an MRP session tunneled over AirPlay."""

import asyncio
import logging
from typing import Callable, Optional, Set

from pyatv import exceptions
from pyatv.auth.hap_srp import SRPAuthHandler
from pyatv.const import PowerState, Protocol
from pyatv.core import Core, MutableService, SetupData, StateMessage, UpdatedState
from pyatv.interface import DeviceListener
from pyatv.protocols import mrp
from pyatv.protocols.airplay.ap2_session import AP2Session
from pyatv.protocols.airplay.mrp_connection import AirPlayMrpConnection
from pyatv.protocols.mrp.protocol import MrpProtocol
from pyatv.support import async_timeout
from pyatv.support.state_producer import StateProducer

_LOGGER = logging.getLogger(__name__)
CONNECT_TIMEOUT = 30
# Power and MRP arrive over different sockets; correlate reordered notifications
# briefly without masking a later, unrelated transport failure.
SLEEP_CORRELATION_TIMEOUT = 5


class _SessionListener(DeviceListener):
    """Forward failures only while this session is authoritative."""

    def __init__(self, tunnel: "AirPlayMrpTunnel") -> None:
        self.tunnel = tunnel
        self.active = True
        self.producer: StateProducer[DeviceListener] = StateProducer()
        self.producer.listener = self

    def connection_closed(self) -> None:
        """Report unexpected EOF from the current session."""
        if self.active:
            self.tunnel.connection_lost(exceptions.ConnectionLostError("AirPlay EOF"))

    def connection_lost(self, exception: Exception) -> None:
        """Report a transport failure from the current session."""
        if self.active:
            self.tunnel.connection_lost(exception)


class AirPlayMrpTunnel:
    """Retain MRP interfaces while replacing sessions retired by sleep."""

    def __init__(self, core: Core, session_factory: Callable[[], AP2Session]) -> None:
        """Initialize a tunnel and subscribe to Companion power observations."""
        self.core = core
        self._session_factory = session_factory
        self._session = session_factory()
        self._listener = _SessionListener(self)
        self._power_state = PowerState.Unknown
        self._suspended = False
        self._sleep_pending: Optional[asyncio.TimerHandle] = None
        self._closed = False
        self._resume_task: Optional[asyncio.Task] = None
        self._cleanup_tasks: Set[asyncio.Task] = set()
        self._connection = AirPlayMrpConnection(
            self._session, self._listener.producer, self._suspend
        )
        service = core.config.get_service(Protocol.MRP)
        if service is None:
            service = MutableService(None, Protocol.MRP, core.service.port, {})
            core.config.add_service(service)
        self._protocol = MrpProtocol(
            self._connection, SRPAuthHandler(), service, core.settings.info
        )
        self._mrp = mrp.create_with_protocol(
            Core(
                core.loop,
                core.config,
                service,
                core.settings,
                core.device_listener,
                core.session_manager,
                core.takeover,
                core.state_dispatcher.create_copy(Protocol.MRP),
            ),
            self._protocol,
            requires_heatbeat=False,
        )
        core.state_dispatcher.listen_to(
            UpdatedState.Power,
            self._power_changed,
            lambda message: message.protocol == Protocol.Companion,
        )
        self.setup_data = SetupData(
            Protocol.MRP,
            self.connect,
            self.close,
            self._mrp.device_info,
            self._mrp.interfaces,
            self._mrp.features,
        )

    async def connect(self) -> bool:
        """Establish and validate a complete AirPlay and MRP session."""
        try:
            async with async_timeout(CONNECT_TIMEOUT):
                await self._session.connect()
                await self._session.setup_remote_control()
                await self._mrp.connect()
            self._session.start_keep_alive(self._listener.producer)
            return True
        except exceptions.HttpError as ex:
            self._retire_session()
            if ex.status_code == 470:
                raise exceptions.InvalidCredentialsError(
                    "invalid or missing credentials"
                ) from ex
            raise
        except (Exception, asyncio.CancelledError):
            self._retire_session()
            raise

    def _suspend(self) -> None:
        # A power observation alone does not retire MRP. Conversely, without a
        # Companion wake path we must retain the existing failure detection.
        if self._closed or self._suspended:
            return
        if self._power_state != PowerState.Off:
            if self._sleep_pending is None:
                self._sleep_pending = self.core.loop.call_later(
                    SLEEP_CORRELATION_TIMEOUT, self._clear_sleep_pending
                )
            return
        self._clear_sleep_pending()
        _LOGGER.debug("Retiring AirPlay MRP session for standby")
        self._suspended = True
        self._retire_session()
        if self._resume_task:
            self._resume_task.cancel()

    def _retire_session(self) -> None:
        # Canceling feedback invokes its finish callback. Retire the listener
        # first so intentional cleanup cannot close the device facade.
        self._listener.active = False
        self._protocol.suspend()
        self._cleanup_tasks.update(self._session.stop())

    def _clear_sleep_pending(self) -> None:
        if self._sleep_pending is not None:
            self._sleep_pending.cancel()
            self._sleep_pending = None

    def _power_changed(self, message: StateMessage) -> None:
        self._power_state = message.value
        if self._sleep_pending and message.value == PowerState.Off:
            self._suspend()
        self._maybe_resume()

    def _maybe_resume(self) -> None:
        if (
            not self._closed
            and self._suspended
            and self._power_state == PowerState.On
            and self._resume_task is None
        ):
            self._resume_task = asyncio.ensure_future(self._resume())

    async def _resume(self) -> None:
        try:
            if self._cleanup_tasks:
                await asyncio.gather(*self._cleanup_tasks, return_exceptions=True)
                self._cleanup_tasks.clear()
            if self._power_state != PowerState.On:
                return
            # RTSP can answer again after wake while the old MRP stream is dead.
            # Recreate authentication and both channels, then repeat MRP startup.
            self._suspended = False
            self._session = self._session_factory()
            self._listener = _SessionListener(self)
            self._connection.session = self._session
            self._connection.device_listener = self._listener.producer
            await self.connect()
            self._suspended = False
            _LOGGER.debug("AirPlay MRP session restored")
        except asyncio.CancelledError:
            pass
        except Exception as ex:
            self.connection_lost(ex)
        finally:
            self._resume_task = None
            self._maybe_resume()

    def connection_lost(self, exc: Exception) -> None:
        """Report genuine failures, including a failed wake handshake."""
        if not self._closed:
            self._cleanup_tasks.update(self.close())
            self.core.device_listener.listener.connection_lost(exc)

    def close(self) -> Set[asyncio.Task]:
        """Permanently close the tunnel and return all pending cleanup tasks."""
        self._closed = True
        self._clear_sleep_pending()
        self._listener.active = False
        tasks = set(self._cleanup_tasks)
        if self._resume_task and self._resume_task is not asyncio.current_task():
            self._resume_task.cancel()
            tasks.add(self._resume_task)
        tasks.update(self._mrp.close())
        tasks.update(self._session.stop())
        return tasks
