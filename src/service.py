from __future__ import annotations

import multiprocessing as mp
import time
from typing import Optional

# local
from message import Message


class BaseService:
    def __init__(self) -> None:
        self.running = False
        self.message_bus: Optional[tuple[
            mp.Queue[Message], mp.connection.Connection
        ]] = None


    def run(self, *args) -> None:
        print(f"{self.__class__.__name__}: running...")

        self.running = True
        while self.running:
            if self.message_bus and self.message_bus[1].poll():
                _, recv = self.message_bus
                message = recv.recv()
                self.on_message(message)
                continue

            self.process(*args)

        print(f"{self.__class__.__name__}: exited.")


    def process(self, *args) -> None:
        print(f"{self.__class__.__name__}: nothing to process..?")


    def stop(self) -> None:
        print(f"{self.__class__.__name__}: stopping...")

        self.running = False

        if self.message_bus:
            self.message_bus[1].close()


    def publish_message(self, msg: Message) -> None:
        print(f"{self.__class__.__name__}: posting message.")

        if self.message_bus:
            send, _ = self.message_bus
            send.put(msg)


    def on_message(self, msg: Message) -> None:
        print(f"{self.__class__.__name__}: message {msg.name} found.")


    def add_message_bus(
        self,
        send: mp.Queue[Message],
        recv: mp.connection.Connection
    ) -> None:
        print(f"{self.__class__.__name__}: added message bus.")
        self.message_bus = (send, recv)

