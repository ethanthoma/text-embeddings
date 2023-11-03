from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import multiprocessing as mp
import time

# local
from service import BaseService
from event import Message


@dataclass
class QueryParams:
    batch_size: int
    size: int
    index: int          = 0


class QueryService(BaseService):
    BASE_QUERY = """
        SELECT r.review_id, r.text as text
        FROM Yelp_review_text.reviews r
        WHERE r.date > TIMESTAMP "2018-03-23 18:41:10 UTC"
        ORDER BY r.date
    """

    def __init__(self, query_params: QueryParams):
        super().__init__()

        self.query_params = query_params

        self.batch = 0
        self.total_batches = ceil(query_params.size / query_params.batch_size)
        self.index = query_params.index


    def process(self, sql_query_queue: mp.Queue[tuple[int, int, str]]) -> None:
        if not sql_query_queue.empty():
            time.sleep(1)
            return

        size_of_current_batch = min(
            self.query_params.batch_size, 
            self.query_params.size - self.index
        )

        query = self.make_query(self.index, size_of_current_batch)
        sql_query_queue.put((self.index, size_of_current_batch, query))
        self.index += size_of_current_batch

        self.batch += 1
        if self.batch == self.total_batches:
            self.stop()


    def make_query(self, index: int, size: int) -> str:
        """Creates from base query and parameters"""
        print("QueryService: creating query...")

        query = self.BASE_QUERY + f"\nLIMIT {size}\nOFFSET {index};"

        print("QueryService: query created.")

        return query


    def on_message(self, msg: Message) -> None:
        super().on_message(msg)
        if msg == Message.EMBED_EXIT or msg == Message.FETCH_EXIT:
            self.stop()


    def stop(self) -> None:
        super().stop()
        self.publish_message(Message.QUERY_EXIT)

