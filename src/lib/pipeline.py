from typing import Callable, Optional, Tuple

import ctypes
from multiprocessing import Array, Process, Queue
from pymonad.either import Either, Left, Right

from lib.service_wrapper import ServiceWrapper

from inspect import signature
class Pipeline:
    def __init__(
            self, 
            *functions: Tuple[Callable[[Optional[Either]], Optional[Either]]]
        ) -> None:

        self.length = len(functions)

        self.services: List[Callable] = [
            ServiceWrapper(function)
            for function in functions
        ]

        self.set_running_states()

        self.queues         = None
        self.running        = False


    def set_running_states(self):
        self.running_states = Array(ctypes.c_bool, [True] * self.length)

        for i, service in enumerate(self.services):
            def setter(fn, index=i):
                for ind in range(index + 1):
                    self.running_states[ind] = False

            service.set_running_as  = setter
            service.is_running      = lambda index=i: self.running_states[index]


    
    def start(self) -> None:
        self.create_queues()

        self.running = True

        self.processes: List[Process] = []

        for i, service in enumerate(self.services):
            input_queue     = (
                self.queues[i - 1] if i > 0 else None
            )
            output_queue    = (
                self.queues[i] if i < self.length - 1 else None
            )

            process = Process(
                target=service, 
                args=(input_queue, output_queue), 
                daemon=True
            )

            self.processes.append(process)
            process.start()


    def create_queues(self) -> None:
        if not self.queues:
            self.queues: List[Queue] = [
                Queue(1)
                for _ in range(len(self.services))
            ]


    def join(self) -> Either[str, None]:
        if not self.running:
            return Left("Pipeline: no processes running, call start() first")

        for process in self.processes:
            process.join()
        return Right(None)

