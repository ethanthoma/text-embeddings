# if you don't have IAM for gcloud project quotas
import warnings
warnings.filterwarnings("ignore", "Your application has authenticated using end user credentials")


# generic
import random
import time
import logging
import pandas as pd

# async
import argparse
import asyncio
import aiohttp

# fetching data from google
from google.cloud import bigquery
from math import ceil

# fetching embeddings
import openai
import os
from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_not_exception_type,
    wait_random_exponential,
)
import tiktoken

# push to bucket
from gcloud.aio.storage import Storage
from io import StringIO


logging.basicConfig()
logger = logging.getLogger(__name__)


openai.api_key = os.environ["POETRY_OPENAI_API_KEY"]


BUCKET_NAME             = "temp_embeddings"
DATASET_NAME            = "vancouver"
TABLE_NAME              = "review"
TEXT_COLUMN             = "text"

BATCH_SIZE              = 2
DATASET_SIZE            = 5

EMBEDDING_MODEL         = "text-embedding-ada-002"
EMBEDDING_ENCODING      = "cl100k_base"
EMBEDDING_CTX_LENGTH    = 8191


async def main(processes: int, index: int, batch: int, size: int) -> None:
    logger.debug("Initializating...")
    queue = asyncio.Queue() # type: asyncio.Queue[int]

    for n in range(ceil(size / batch)):
        await queue.put((n * batch) + index)

    processors = [
        asyncio.create_task(job(queue, batch)) 
        for _ in range(processes)
    ]
    logger.debug("Created tasks.")

    await queue.join()
    for processor in processors:
        processor.cancel()


async def job(queue: asyncio.queues.Queue, batch_size: int) -> None:
    async with aiohttp.ClientSession() as session:
        bq = bigquery.Client()
        storage = Storage(session=session)

        while True:
            logger.debug("Job start")
            index = await queue.get()
            start = time.perf_counter()

            query = make_query(index, batch_size)

            df = await bq_query(bq, query)

            df = await embed_text(df, TEXT_COLUMN)
            print(df.size, df.index[1])

            await upload_df_to_storage(storage, df, f"batch-index_{df.index[1]}-size_{df.size}")

            end = time.perf_counter() - start
            logger.debug(f"Job complete: index {index} took {end:0.2f} seconds.")

            queue.task_done()


async def embed_text(df: pd.DataFrame, text_column: str = TEXT_COLUMN) -> pd.DataFrame:
    embeddings = await fetch_embeddings(df, text_column)
    print(type(embeddings), type(embeddings[1]))
    embeddings = [pd.Series(embedding) for embedding in embeddings]

    df = df.drop(text_column, axis=1)
    df = pd.merge(df, embeddings, left_index=True, right_index=True)

    return df


def make_query(
        index: int, 
        size: int, 
        dataset_name: str = DATASET_NAME,
        table_name: str = TABLE_NAME
    ) -> str:
    """A query to fetch a batch of data"""
    logger.debug("Creating query...")
    return f"""
        SELECT review_id as id, text
        FROM {dataset_name}.{table_name}
        ORDER BY date
        LIMIT {size}
        OFFSET {index};
    """


async def bq_query(bq: bigquery.client.Client, query: str) -> pd.DataFrame:
    """Async BigQuery.query()"""
    logger.debug("Sending query...")

    loop = asyncio.get_event_loop()

    future = loop.create_future()

    def callback(job):
        result = job.result().to_dataframe()
        loop.call_soon_threadsafe(future.set_result, result)

    job = await loop.run_in_executor(None, bq.query, query)
    job.add_done_callback(callback)

    return await future


# TODO: minibatch, compare w/ context limit, split 
async def fetch_embeddings(df, column_to_embed: str = "text") -> list:
    """Fetch embeddings from a row generator function"""
    logger.debug("Fetching OpenAI embeddings...")

    loop = asyncio.get_event_loop()

    start = time.perf_counter()

    encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)

    all_embeddings = []
    current_batch = []
    current_token_count = 0

    token_count = (
        df[column_to_embed]
        .apply(lambda x: len(encoding.encode(x)))
        .sum()
    )

    print(df[column_to_embed].to_list())

    embeddings = await loop.run_in_executor(
        None, 
        get_embedding, 
        df[column_to_embed].to_list()
    )

    end = time.perf_counter() - start
    logger.debug(
        "Fetching embeddings took "   
        f"{end:0.2f} seconds for {token_count} tokens."
    )
    
    return embeddings


@retry(
    wait=wait_random_exponential(min=1, max=20), 
    stop=stop_after_attempt(6), 
    retry=retry_if_not_exception_type(openai.InvalidRequestError)
)
def get_embedding(text_or_tokens: list, model: str = EMBEDDING_MODEL) -> list:
    """Get embeddings of a list of text/tokens with backoff from OpenAI"""
    return openai.Embedding.create(
        input=text_or_tokens, 
        model=model
    )["data"][0]["embedding"]


async def upload_df_to_storage(
        storage, 
        batch: pd.DataFrame, 
        filename: str
    ) -> None:
    logger.debug("upload_df_to_storage")
    start = time.perf_counter()

    print(type(storage))

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    await storage.upload(BUCKET_NAME, filename, csv_data)

    end = time.perf_counter() - start
    logger.debug(f"upload_df_to_storage sleeping for {end:0.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--processes", type=int, default=1)
    parser.add_argument("-i", "--index", type=int, default=0)
    parser.add_argument("-b", "--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("-s", "--size", type=int, default=DATASET_SIZE)
    ns = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    start = time.perf_counter()

    asyncio.run(main(**ns.__dict__))

    end = time.perf_counter() - start
    print(f"Program finished in {end:0.2f} seconds.")

