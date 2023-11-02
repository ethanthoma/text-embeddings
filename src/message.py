from enum import Enum
import multiprocessing as mp

# local
from service import BaseService


class Message(Enum):
    QUERY_EXIT = "QueryServiceExit"
    FETCH_EXIT = "FetchServiceExit"
    EMBED_EXIT = "EmbedServiceExit"
    STORE_EXIT = "StoreServiceExit"


class MessageBus():
    def __init__(self) -> None:
        print(f"MessageBus: init.")
        self.queue: mp.Queue[Message] = mp.Queue()
        self.connections: list[mp.connection.Connection] = []


    def run(self):
        print(f"MessageBus: running...")
        self.running = True
        while self.running:
            msg = self.queue.get()
            
            self.publish(msg)


    def publish(self, msg: Message):
        print(f"MessageBus: publishing to services...")
        for conn in self.connections:
            conn.send(msg)


    def register(self, service: BaseService):
        send, recieve = mp.Pipe()
        service.add_message_bus(self.queue, recieve)
        self.connections.append(send)


    def stop(self):
        print(f"MessageBus: stopping...")
        self.running = False

        for conn in self.connections:
            conn.close()

