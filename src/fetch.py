from __future__ import annotations

import warnings
warnings.filterwarnings(
    "ignore", 
    "Your application has authenticated using end user credentials"
)

from dataclasses import dataclass
from google.cloud import bigquery as bq # type: ignore
import multiprocessing as mp
import os
import pandas as pd
from pathlib import Path
import time

# local
from service import BaseService
from event import Message


@dataclass
class FetchParams:
    client: bq.Client
    cache_dir: str      = "./cache"


class FetchService(BaseService):
    def __init__(self, fetch_params: FetchParams):
        super().__init__()
        self.fetch_params = fetch_params
        self.on_empty = lambda: time.sleep(1)

        Path(fetch_params.cache_dir).mkdir(parents=True, exist_ok=True)


    def process(
        self, 
        sql_query_queue: mp.Queue[tuple[int, int, str]],
        filename_queue: mp.Queue[str]
    ) -> None:
        if sql_query_queue.empty():
            self.on_empty()
            return

        if not filename_queue.empty():
            time.sleep(1)
            return

        index, size, sql_query = sql_query_queue.get()
        df = self.perform_query(sql_query)

        if len(df.index) == 0:
            self.stop()
            return

        filename = f"{index}_{index + size}.csv"
        save_path = f"{self.fetch_params.cache_dir}/{filename}"
        self.save_dataframe_to_file(df, save_path)
        filename_queue.put(filename)


    def perform_query(self, sql_query: str) -> pd.DataFrame:
        print(f"FetchService: querying...")
        client = self.fetch_params.client
        job = client.query(sql_query)
        df = job.result().to_dataframe()
        return df


    def save_dataframe_to_file(self, df: pd.DataFrame, filename: str) -> None:
        print(f"FetchService: saving to {filename}...")
        df.to_csv(filename, index=False)


    def on_message(self, msg: Message) -> None:
        super().on_message(msg)

        if msg == Message.QUERY_EXIT:
            self.on_empty = self.stop
        elif msg == Message.EMBED_EXIT:
            self.stop()


    def stop(self) -> None:
        super().stop()
        self.publish_message(Message.FETCH_EXIT)

