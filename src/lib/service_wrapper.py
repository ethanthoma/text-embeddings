from __future__ import annotations

from typing import Callable, Optional

from dataclasses import astuple
from multiprocessing import Queue
from pymonad.either import Either
from time import sleep


class ServiceWrapper:
    def __init__(
        self,
        func: Callable[..., Either]
        ) -> None:
        self.func       = func
        self.running    = True


    def set_running_as(self, conditional: Callable[bool]) -> None:
        self.conditional = conditional


    def is_running(self) -> bool:
        return self.conditional


    def __call__(
        self,
        input_queue:    Optional[Queue[Either]] = None, 
        output_queue:   Optional[Queue[Either]] = None
        ) -> None:
        while self.is_running():
            if output_queue and output_queue.qsize() > 0: 
                sleep(1)
                continue

            if input_queue and input_queue.empty():
                sleep(1)
                continue

            if input_queue:
                input = input_queue.get()
                if input.is_left():
                    self.set_running_as(False)
                elif input.is_right():
                    args = astuple(input.value)
                    output = self.func(*args)
            else:
                output = self.func()

            if output:
                if output_queue:
                    output_queue.put(output)
                if output.is_left():
                    self.set_running_as(False)

