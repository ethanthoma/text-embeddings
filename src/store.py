from __future__ import annotations

from dataclasses import dataclass
from google.cloud import storage # type: ignore
from io import BytesIO
import multiprocessing as mp
import os
import pandas as pd
import time
from typing import Callable

# local
from service import BaseService
from event import Message


@dataclass
class StoreParams:
    bucket_name: str
    embedded_file_dir: str      = "./cache"


class StoreService(BaseService):
    def __init__(self, store_params: StoreParams):
        super().__init__()

        self.store_params = store_params

        self.on_empty: Callable = lambda: None


    def process(self, embed_filename_queue: mp.Queue[str]) -> None:
        if embed_filename_queue.empty():
            time.sleep(1)
            self.on_empty()
            return

        filename = embed_filename_queue.get()
        path = f"{self.store_params.embedded_file_dir}/{filename}"
        df = pd.read_csv(path)

        file_obj = self.dataframe_to_bytesio(df)
        success = self.upload_to_gcs(file_obj, filename)


    def dataframe_to_bytesio(self, df: pd.DataFrame) -> BytesIO:
        """Convert dataframe to bytes"""
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer


    def upload_to_gcs(self, file_obj: BytesIO, filename: str) -> bool:
        """Streams bytes of the dataframe to Google Cloud Storage"""
        start = time.perf_counter()

        storage_client = storage.Client()
        bucket = storage_client.bucket(self.store_params.bucket_name)

        blob = bucket.blob(filename)
        file_obj.seek(0)

        try:
            blob.upload_from_file(file_obj)

            end = time.perf_counter() - start
            print(
                f"StoreService: Uploaded {filename} "
                f"to GCS in {end:0.2f} seconds."
            )
            return True
        except Exception as e:
            end = time.perf_counter() - start
            print(
                f"StoreService: Failed to upload {filename} "
                f"to GCS after {end:0.2f} seconds."
            )
            return False


    def on_message(self, msg: Message) -> None:
        super().on_message(msg)
        if msg == Message.EMBED_EXIT:
            self.on_empty = self.stop


    def stop(self) -> None:
        super().stop()
        self.publish_message(Message.STORE_EXIT)


