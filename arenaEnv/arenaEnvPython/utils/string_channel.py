from mlagents_envs.side_channel.side_channel import (SideChannel, IncomingMessage, OutgoingMessage)
import uuid


class StringChannel(SideChannel):

    def __init__(self, logger) -> None:
        super().__init__(uuid.UUID("621f0a93-4f87-11ea-a6bf-722f4387d1f7"))
        self.logger = logger

    def on_message_received(self, msg: IncomingMessage) -> None:
        if self.logger is not None:
            self.logger.info(f"MESSAGE FROM UNITY ENV: {msg.read_string()}")
        else:
            print(f"MESSAGE FROM UNITY ENV: {msg.read_string()}")

    def send_string(self, data: str) -> None:
        msg = OutgoingMessage()
        msg.write_string(data)
        super().queue_message_to_send(msg)
