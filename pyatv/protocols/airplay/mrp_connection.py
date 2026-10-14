"""MRP connection implemented as a channel/stream over AirPlay."""

import logging
from typing import Callable, Optional

from pyatv import exceptions
from pyatv.protocols.airplay.ap2_session import AP2Session
from pyatv.protocols.airplay.channels import DataStreamChannel, DataStreamListener
from pyatv.protocols.mrp import protobuf
from pyatv.protocols.mrp.connection import AbstractMrpConnection
from pyatv.protocols.mrp.protobuf.ProtocolMessage_pb2 import ErrorCode
from pyatv.support import log_protobuf
from pyatv.support.state_producer import StateProducer

_LOGGER = logging.getLogger(__name__)


class AirPlayMrpConnection(AbstractMrpConnection, DataStreamListener):
    """Transparent connection/channel for transporting MRP messages."""

    def __init__(
        self,
        session: AP2Session,
        device_listener: Optional[StateProducer] = None,
        on_sleep: Optional[Callable[[], None]] = None,
    ):
        """Initialize a new MrpConnection."""
        super().__init__()
        self.session = session
        self.data_channel: Optional[DataStreamChannel] = None
        self.device_listener = device_listener
        self.on_sleep = on_sleep
        self._channel_listener: Optional[_ChannelListener] = None

    def connection_lost(self, exc: Optional[Exception]) -> None:
        """Device connection was dropped."""
        self.handle_connection_lost(exc)

    async def connect(self) -> None:
        """Connect to device."""
        if self.session.data_channel is None:
            raise exceptions.InvalidStateError("remote control channel not connected")

        self.data_channel = self.session.data_channel
        self._channel_listener = _ChannelListener(self, self.data_channel)
        self.data_channel.listener = self._channel_listener

    def enable_encryption(self, output_key: bytes, input_key: bytes) -> None:
        """Enable encryption with the specified keys."""

    @property
    def connected(self) -> bool:
        """If a connection is open or not."""
        return self.data_channel is not None

    def close(self) -> None:
        """Close connection to device."""
        channel = self.data_channel
        self.data_channel = None
        self._channel_listener = None
        if channel is not None:
            _LOGGER.debug("Closing connection")
            channel.close()

    def send(self, message: protobuf.ProtocolMessage) -> None:
        """Send protobuf message to device."""
        if self.data_channel is None:
            raise exceptions.InvalidStateError("remote control channel not connected")
        self.data_channel.send_protobuf(message)
        log_protobuf(_LOGGER, ">> Send: Protobuf", message)

    def handle_protobuf(self, message: protobuf.ProtocolMessage) -> None:
        """Handle incoming protobuf message."""
        log_protobuf(_LOGGER, "<< Receive: Protobuf", message)

        self.listener.message_received(message, None)  # pylint: disable=no-member

        if (
            message.type == protobuf.SET_CONNECTION_STATE_MESSAGE
            and protobuf.extract_inner(message).state
            == protobuf.SetConnectionStateMessage.Disconnected
        ):
            if message.errorCode == ErrorCode.TheDeviceIsGoingToSleep:
                if self.on_sleep:
                    self.on_sleep()
            else:
                self.handle_connection_lost(
                    exceptions.ConnectionLostError(
                        f"MRP disconnected (error {message.errorCode})"
                    )
                )

    def handle_connection_lost(self, exc: Optional[Exception]) -> None:
        """Device connection was dropped."""
        _LOGGER.debug("Disconnected from device: %s", exc)

        if self.device_listener:
            if exc is None:
                self.device_listener.listener.connection_closed()
            else:
                self.device_listener.listener.connection_lost(exc)


class _ChannelListener(DataStreamListener):
    """Ignore messages and EOF from a retired data channel."""

    def __init__(self, connection: AirPlayMrpConnection, channel: DataStreamChannel):
        self.connection = connection
        self.channel = channel

    def handle_protobuf(self, message: protobuf.ProtocolMessage) -> None:
        if self.connection.data_channel is self.channel:
            self.connection.handle_protobuf(message)

    def handle_connection_lost(self, exc: Optional[Exception]) -> None:
        if self.connection.data_channel is self.channel:
            self.connection.handle_connection_lost(exc)
