from __future__ import annotations

from typing import Callable, List
from dataclasses import dataclass

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import multiprocessing as mp
import numpy as np
import openai
import os
import pandas as pd
from ratemate import RateLimit
from tenacity import retry, wait_random_exponential, stop_after_attempt
import tiktoken
import time
from threading import Lock

# local
from service import BaseService
from message import Message


@dataclass
class EmbedParams:
    openai_key: str
    requests_so_far: int        = 0
    max_workers: int            = 5
    text_column: str            = "text"
    requests_per_day_limit: int = 10_000
    requests_per_min_limit: int = 500
    embedding_encoding: str     = "cl100k_base"
    embedding_model: str        = "text-embedding-ada-002"
    context_size: int           = 8_191
    embedding_size: int         = 1_536
    cache_dir: str              = "./cache"
    embedded_file_dir: str      = "./cache"
    embedded_file_suffix: str   = "embedded"
    unembedded_file_dir: str    = "./cache"
    unembedded_file_suffix: str = "unembedded"


class EmbedService(BaseService):
    def __init__(self, embed_params: EmbedParams) -> None:
        super().__init__()
        self.embed_params = embed_params

        self.on_empty: Callable = lambda: time.sleep(1)
        self.api_calls: int = embed_params.requests_so_far
        self.api_lock = Lock()
        self.rate_limit = RateLimit(
            max_count=self.embed_params.requests_per_min_limit,
            per=60
        )

        openai.api_key = embed_params.openai_key

    def process(
        self,
        raw_filename_queue: mp.Queue[str],
        embed_filename_queue: mp.Queue[str]
    ) -> None:
        if raw_filename_queue.empty():
            self.on_empty()
            return

        filename = raw_filename_queue.get()
        df = pd.read_csv(filename)

        df["n_tokens"] = self.token_count(df[self.embed_params.text_column])
        chunks = self.chunk_dataframe(df)
        chunks = self.parallel_embed_chunks(chunks)

        basename = os.path.basename(os.path.normpath(filename))
        embedded_df = pd.concat(chunks)

        if len(embedded_df.index) != 0:
            embedded_filename = (
                f"{self.embed_params.embedded_file_dir}/"
                f"{self.embed_params.embedded_file_suffix}_"
                f"{basename}"
            )
            embedded_df.to_csv(embedded_filename)
            embed_filename_queue.put(embedded_filename)

        if df.shape[0] != embedded_df.shape[0]:
            unembedded_indices = df.index.difference(embedded_df.index)
            unembedded_df = df.loc[unembedded_indices]
            unembedded_filename = (
                f"{self.embed_params.unembedded_file_dir}/"
                f"{self.embed_params.unembedded_file_suffix}_"
                f"{basename}"
            )
            unembedded_df.to_csv(unembedded_filename)

        os.remove(filename)


    def cache_embedded_chunks(
        self,
        chunks: list[pd.DataFrame], 
        filename: str
    ) -> None:
        print("EmbedService: caching embedded chunks...")
        for chunk in chunks:
            indices = chunk.indices
            embeddings = chunk.embeddings

        df: pd.DataFrame
        df = pd.merge(df, embeddings, left_index=True, right_index=True)
        df.to_csv()


    def write_unembbeded_chunks(
        self, df: pd.DataFrame, 
        chunks: list[pd.DataFrame], 
        filename: str
    ) -> None:
        print("EmbedService: writing unembedded chunks...")
        embeddings = pd.concat([chunk.embeddings for chunk in chunks])
        df = pd.merge(df, embeddings, left_index=True, right_index=True)
        df.to_csv()
        

    def token_count(
        self, 
        text: pd.Series
    ) -> pd.Series:
        """Generate token length series from text series"""
        print("EmbedService: generating token lengths...")
        encoding = tiktoken.get_encoding(self.embed_params.embedding_encoding)

        return text.apply(lambda x: len(encoding.encode(x)))


    def chunk_dataframe(
        self,
        df: pd.DataFrame
    ) -> list[pd.DataFrame]:
        """Splits dataframe into chunks that fit within the context token 
        length"""
        print("EmbedService: chunking df...")
        df["group"] = self.chunk_ids_from_token_length(df["n_tokens"])

        chunks: list[pd.DataFrame] = []
        for _, chunk in df.groupby('group'):
            chunk = chunk.drop(['group'], axis=1)
            chunks.append(chunk)
            
        return chunks


    def chunk_ids_from_token_length(
        self,
        token_counts: pd.Series[int]
    ) -> pd.Series:
        """Generates a series that represents which group each text belongs to 
        such that each group is under the context size limit"""
        print("EmbedService: generating chunk splits...")

        group_identifiers = np.zeros(len(token_counts))

        current_group_token_count = 0
        current_group = 0

        def should_create_new_group(next_token_count) -> bool:
            return next_token_count > self.embed_params.context_size

        for index, token_count in enumerate(token_counts):
            if should_create_new_group(current_group_token_count + token_count):
                current_group += 1
                current_group_token_count = 0
            
            current_group_token_count += token_count
            group_identifiers[index] = current_group

        return pd.Series(group_identifiers, dtype=int)


    def parallel_embed_chunks(
        self, 
        chunks: List[pd.DataFrame]
    ) -> List[pd.DataFrame]:
        print("EmbedService: parallel embedding chunks...")
        results: List[pd.DataFrame]

        with ThreadPoolExecutor(
            max_workers=self.embed_params.max_workers
        ) as executor:
            results = list(executor.map(self.embed_chunk, chunks))

        return results


    def embed_chunk(
        self, 
        chunk: pd.DataFrame 
    ) -> pd.DataFrame:
        """Get embeddings of a chunk of text from OpenAI"""
        print("EmbedService: embedding one chunk...")
        with self.api_lock:
            if self.api_calls >= self.embed_params.requests_per_day_limit:
                return pd.DataFrame()

        text = chunk[self.embed_params.text_column]
        
        embed_response = self.api_call(text)

        embeddings_df = pd.DataFrame(
            np.nan, 
            index=range(len(text)), 
            columns=[
                f'dim_{i}' for i in range(self.embed_params.embedding_size)
            ]
        )

        for obj in embed_response:
            index = obj['index']
            embedding = obj['embedding']
            embeddings_df.iloc[index] = embedding

        chunk = pd.concat([chunk, embeddings_df], axis=1)
        chunk = chunk.drop([self.embed_params.text_column], axis=1)

        return chunk


    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6)
    )
    def api_call(
        self, 
        text: pd.Series[str]
    ) -> list[openai.openai_object.OpenAIObject]:
        self.rate_limit.wait()
        print("EmbedService: api call with retry...")

        self.increase_api_count()

        embed_response = openai.Embedding.create(
            input=text.to_list(), 
            model=self.embed_params.embedding_model
        )["data"]

        return embed_response


    def increase_api_count(self):
        with self.api_lock:
            self.api_calls += 1


    def on_message(self, msg: Message) -> None:
        super().on_message(msg)
        if msg.name == "FETCH_EXIT":
            self.on_empty = self.stop


    def stop(self) -> None:
        super().stop()
        self.publish_message(Message.EMBED_EXIT)

