# if you don't have IAM for gcloud project quotas
import warnings
warnings.filterwarnings("ignore", "Your application has authenticated using end user credentials")


# generic
import logging
import numpy as np
import pandas as pd
import random
import time

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
DATASET_NAME            = "Yelp_review_text"
TABLE_NAME              = "reviews"
TEXT_COLUMN             = "text"

EMBEDDING_CTX_LENGTH    = 8191
EMBEDDING_SPACE         = 1536
EMBEDDING_ENCODING      = "cl100k_base"
EMBEDDING_MODEL         = "text-embedding-ada-002"


async def main(
        processes: int, 
        index: int, 
        batch: int, 
        size: int, 
        verbose: bool
    ) -> None:
    logger.setLevel(
        logging.DEBUG if verbose else logging.INFO
    )

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
            index = await queue.get()
            logger.info(
                f"Job start: at index {index} "
                f"with estimated size {batch_size}."
            )
            start = time.perf_counter()

            query = make_query(index, batch_size)

            df = await bq_query(bq, query)

            df = await embed_text(
                df, 
                TEXT_COLUMN,
                EMBEDDING_ENCODING,
                EMBEDDING_MODEL,
                EMBEDDING_CTX_LENGTH,
                EMBEDDING_SPACE
            )

            await upload_df_to_storage(
                storage, 
                df, 
                f"batch-index_{index}_to_{index + df.shape[0] - 1}"
            )

            end = time.perf_counter() - start
            logger.info(
                f"Job complete: took {end:0.2f} " 
                f"seconds for index {index} to {index + df.shape[0] - 1}"
            )

            queue.task_done()


def make_query(index: int, size: int) -> str:
    """A query to fetch a batch of data"""
    logger.debug("Creating query...")
    return f"""
        SELECT r.review_id, r.text
        FROM Yelp_review_text.reviews r
        JOIN Yelp_review_text.business b 
        ON r.business_id = b.business_id
        WHERE b.state != 'BC'
        ORDER BY r.date
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


async def embed_text(
    df: pd.DataFrame,
    text_column: str,
    embedding_encoding: str,
    embedding_model: str,
    context_size: int,
    embedding_size: int
) -> pd.DataFrame:
    """Embeds the text column of dataframe, adds it at columns, and adds the 
    token count"""
    logger.debug("Embedding text...")

    df["n_tokens"] = token_length_of_text(
        df[text_column],
        embedding_encoding
    )

    df['group'] = label_text_series_into_subgroups_within_context_size(
        df["n_tokens"],
        context_size
    )

    grouped = df.groupby('group')

    tasks = []
    for _, group in grouped:
        task = asyncio.create_task(
            batch_get_embeddings_from_series_as_df(
                group[text_column], 
                embedding_model,
                embedding_size
            )
        )
        tasks.append(task)

    embeddings = pd.concat(await asyncio.gather(*tasks))

    df = df.drop([text_column, 'group'], axis=1)
    df = pd.merge(df, embeddings, left_index=True, right_index=True)

    return df


def token_length_of_text(
        text: pd.Series,
        embedding_encoding: str
    ) -> pd.Series:
    """Generate token length series from text series"""
    encoding = tiktoken.get_encoding(embedding_encoding)

    return text.apply(lambda x: len(encoding.encode(x)))


def label_text_series_into_subgroups_within_context_size(
        token_count_series: pd.Series,
        context_size: int
    ) -> pd.Series:
    """Generates a series that represents which group each text belongs to such
    that each group is under the context size limit"""

    group_identifiers = np.zeros(len(token_count_series))

    current_group_token_count = 0
    current_group = 0
    for index, token_count in enumerate(token_count_series):
        if current_group_token_count + token_count > context_size:
            current_group += 1
            current_group_token_count = 0
        
        current_group_token_count += token_count
        group_identifiers[index] = current_group

    return pd.Series(group_identifiers, dtype=int)
        

async def batch_get_embeddings_from_series_as_df(
        text: pd.Series,
        embedding_model: str,
        embedding_size: int
    ) -> pd.DataFrame:
    """Fetch OpenAI embeddings..."""
    logger.debug("Fetching OpenAI embeddings...")

    loop = asyncio.get_event_loop()

    start = time.perf_counter()

    embeddings_df = await loop.run_in_executor(
        None, 
        lambda args: get_batch_embedding_of_text(**args), 
        { 
            'text':             text, 
            'embedding_model':  embedding_model, 
            'embedding_size':   embedding_size 
        }
    )

    end = time.perf_counter() - start
    logger.debug(
        f"Fetching {len(text)} embeddings took {end:0.2f} seconds."
    )
    
    return embeddings_df


@retry(
    wait=wait_random_exponential(min=1, max=20), 
    stop=stop_after_attempt(6), 
    retry=retry_if_not_exception_type(openai.InvalidRequestError)
)
def get_batch_embedding_of_text(
        text: pd.Series, 
        embedding_model: str, 
        embedding_size: int
    ) -> pd.DataFrame:
    """Get embeddings of a list of text/tokens with backoff from OpenAI"""
    embed_response = openai.Embedding.create(
        input=text.to_list(), 
        model=embedding_model
    )["data"]

    embeddings_df = pd.DataFrame(
        np.nan, 
        index=range(len(text)), 
        columns=[f'dim_{i}' for i in range(embedding_size)]
    )

    for obj in embed_response:
        index = obj['index']
        embedding = obj['embedding']
        embeddings_df.iloc[index] = embedding 

    return embeddings_df


async def upload_df_to_storage(
        storage: Storage, 
        df: pd.DataFrame, 
        filename: str
    ) -> None:
    """Uploads a dataframe to Google Cloud Storage"""
    logger.debug("Uploading to storage...")
    start = time.perf_counter()

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    await storage.upload(BUCKET_NAME, filename, csv_data)

    end = time.perf_counter() - start
    logger.debug(f"Uploading took {end:0.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--processes", type=int, default=1, 
        help="Number of workers"
    )
    parser.add_argument(
        "-i", "--index", type=int, default=0, 
        help="Starting index in the dataset"
    )
    parser.add_argument(
        "-b", "--batch", type=int, default=2, 
        help="Size of batches per each worker"
    )
    parser.add_argument(
        "-s", "--size", type=int, default=5, 
        help="Size of the dataset"
    )
    parser.add_argument(
        "-v", "--verbose", action='store_true', 
        help="Add to make verbose"
    )
    ns = parser.parse_args()

    logger = logging.getLogger(__name__)

    start = time.perf_counter()

    asyncio.run(main(**ns.__dict__))

    end = time.perf_counter() - start
    print(f"Program finished in {end:0.2f} seconds.")

