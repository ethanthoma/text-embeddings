from collections import deque
from threading import Lock
import time

class TokenRateLimiter:
    def __init__(self, max_tokens_per_minute):
        self.max_tokens_per_minute = max_tokens_per_minute
        self.tokens = deque()
        self.lock = Lock()

    def add_and_wait_if_needed(self, token_count):
        with self.lock:
            while not self.space_in_limit_for(token_count):
                self.expire_old_tokens()

                time.sleep(0.12)

            self.tokens.append((time.time(), token_count))


    def expire_old_tokens(self) -> None:
        current_time = time.time()

        while self.tokens and current_time - self.tokens[0][0] > 60:
            self.tokens.popleft()


    def space_in_limit_for(self, token_count: int) -> bool:
        total_tokens: int = self.total_tokens()
        return total_tokens + token_count <= self.max_tokens_per_minute


    def total_tokens(self) -> int:
        return sum(token for _, token in self.tokens)

    
