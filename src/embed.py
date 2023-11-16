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
from event import Message
from token_rate_limiter import TokenRateLimiter


@dataclass
class EmbedParams:
    openai_key: str
    requests_so_far: int        = 0
    max_workers: int            = 5
    text_column: str            = "text"
    requests_per_day_limit: int = -1 
    requests_per_min_limit: int = 5_000 
    tokens_per_min_limit: int   = 5_000_000
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

        self.on_empty: Callable = lambda: None

        self.api_calls: int = embed_params.requests_so_far
        self.api_lock = Lock()
        self.rate_limit = RateLimit(
            max_count=embed_params.requests_per_min_limit,
            per=60
        )
        openai.api_key = embed_params.openai_key

        self.tokens_embedded: int = 0
        self.token_lock = Lock()

        self.token_rate_limiter = TokenRateLimiter(
            max_tokens_per_minute=embed_params.tokens_per_min_limit
        )


    def process(
        self,
        raw_filename_queue: mp.Queue[str],
        embed_filename_queue: mp.Queue[str]
    ) -> None:
        if raw_filename_queue.empty():
            time.sleep(1)
            self.on_empty()
            return

        filename = raw_filename_queue.get()
        path = f"{self.embed_params.cache_dir}/{filename}"
        df = pd.read_csv(path)
        df[self.embed_params.text_column] = df[self.embed_params.text_column].astype(str)

        df["n_tokens"] = self.token_count(df[self.embed_params.text_column])
        chunks = self.chunk_dataframe(df)
        print(f"EmbedService: embedding {len(chunks)} chunks")
        chunks = self.parallel_embed_chunks(chunks)

        embedded_df = pd.concat(chunks, ignore_index=True)

        if len(embedded_df.index) != 0:
            embedded_filename = (
                    f"{self.embed_params.embedded_file_suffix}_"
                    f"{filename}"
            )
            self.save_embeddings(embedded_df, embedded_filename)
            embed_filename_queue.put(embedded_filename)

        self.save_unembedded_data(df, embedded_df, filename)


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

        start = time.perf_counter()
        start_amt = self.tokens_embedded

        with ThreadPoolExecutor(
            max_workers=self.embed_params.max_workers
        ) as executor:
            results = list(executor.map(self.embed_chunk, chunks))

        end = time.perf_counter() - start
        total_amt = self.tokens_embedded - start_amt

        print(f"EmbedService: embedded {total_amt} tokens in {end:0.2f} seconds")

        return results


    def embed_chunk(
        self, 
        chunk: pd.DataFrame 
    ) -> pd.DataFrame:
        """Get embeddings of a chunk of text from OpenAI"""
        with self.api_lock:
            if (self.embed_params.requests_per_day_limit != -1 and
                self.api_calls >= self.embed_params.requests_per_day_limit):
                print("EmbedService: max api calls reached.")
                return pd.DataFrame()

        chunk = chunk.reset_index(drop=True)
        text = chunk[self.embed_params.text_column]
        token_count = sum(chunk['n_tokens'])
        embed_response = self.api_call(
            text = text, 
            token_count = token_count
        )

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

        if len(chunk.index) != 0:
            with self.token_lock:
                self.tokens_embedded += sum(chunk['n_tokens'])

        return chunk


    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6)
    )
    def api_call(
        self, 
        text: pd.Series[str],
        token_count: int
    ) -> list[openai.openai_object.OpenAIObject]:
        self.token_rate_limiter.add_and_wait_if_needed(token_count)
        self.rate_limit.wait()

        print(f"EmbedService: api call #{self.api_calls + 1}.")

        self.increase_api_count()
        response = openai.Embedding.create(
            input=text.to_list(), 
            model=self.embed_params.embedding_model
        )

        return response["data"]


    def increase_api_count(self):
        with self.api_lock:
            self.api_calls += 1


    def save_embeddings(self, df: pd.DataFrame, filename: str) -> None:
        print("EmbedService: saving embedded df...")
        path = f"{self.embed_params.embedded_file_dir}/{filename}"
        df.to_csv(path, index=False)


    def save_unembedded_data(
        self,
        df: pd.DataFrame, 
        embedded_df: pd.DataFrame,
        filename: str
    ) -> None:
        if df.shape[0] != embedded_df.shape[0]:
            print("EmbedService: saving unembedded df...")
            indices = df.index.difference(embedded_df.index)
            unembedded_df = df.loc[indices]
            path = (
                f"{self.embed_params.unembedded_file_dir}/"
                f"{self.embed_params.unembedded_file_suffix}_"
                f"{filename}"
            )
            unembedded_df.to_csv(path, index=False)


    def on_message(self, msg: Message) -> None:
        super().on_message(msg)
        if msg.name == "FETCH_EXIT":
            self.on_empty = self.stop


    def stop(self) -> None:
        super().stop()
        self.publish_message(Message.EMBED_EXIT)

