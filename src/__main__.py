# if you don't have IAM for gcloud project quotas
import warnings
warnings.filterwarnings(
    "ignore", 
    "Your application has authenticated using end user credentials"
)


# generic
import argparse
import logging
import numpy as np
import pandas as pd
import time
from dataclasses import dataclass

# async
import aiofiles
import aiohttp
import asyncio

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
from google.cloud import storage
from io import BytesIO


# local imports
from configure_logger import configure_logger


# argument defaults
ARG_DEFAULT_INDEX       = 0
ARG_DEFAULT_BATCH       = 10
ARG_DEFAULT_SIZE        = 20


openai.api_key = os.environ["POETRY_OPENAI_API_KEY"]


BASE_QUERY = """
    SELECT r.review_id, r.text as text
    FROM Yelp_review_text.reviews r
    WHERE r.date > TIMESTAMP "2015-04-20 06:24:29 UTC"
    ORDER BY r.date
"""

EMBEDDING_CTX_LENGTH    = 8191
EMBEDDING_SPACE         = 1536
EMBEDDING_ENCODING      = "cl100k_base"
EMBEDDING_MODEL         = "text-embedding-ada-002"
OPENAI_RPD_LIMIT        = 10_000
TEXT_COLUMN             = "text"

BUCKET_NAME             = "temp_embeddings"


logger = logging.getLogger(__name__)

async def main(
    index: int, 
    batch: int, 
    size: int, 
    requests: int
) -> None:
    logger.debug("Initializating BigQuery API call tasks.")

    bq_api_calls: asyncio.Queue[int] = asyncio.Queue()
    for n in range(ceil(size / batch)):
        bq_api_calls.put_nowait((n * batch) + index)

    gcs_reqs: list[asyncio.Task] = []

    logger.debug("Starting tasks...")
    asyncio.create_task(
        job(bq_api_calls, gcs_reqs, batch, requests)
    )

    await bq_api_calls.join()

    await asyncio.gather(*gcs_reqs)


async def job(
    bq_api_calls: asyncio.Queue[int], 
    gcs_reqs: list,
    batch_size: int,
    open_ai_reqs_so_far: int
) -> None:
    bq = bigquery.Client()
    
    while open_ai_reqs_so_far < OPENAI_RPD_LIMIT:
        index = await bq_api_calls.get()
        logger.info(
            f"Job #{index} start: "
            f"with estimated size {batch_size}."
        )
        start = time.perf_counter()

        # fetch query data
        query_params = QueryParams(bq, index, batch_size)
        df = await perform_query(query_params)

        embed_params = EmbedParams(df, open_ai_reqs_so_far)
        df, open_ai_reqs_so_far = await embed_text(
            df, 
            open_ai_reqs_so_far,
            embed_params
        )

        # TEMP: adjust naming scheme with adjusted query
        index += 1950000
        filename = f"batch-index_{index}_to_{index + df.shape[0] - 1}.csv"
        save_df_to_file_params = SaveDFToFileParams(df, filename)

        async def _save_with_log():
            await save_df_to_file(save_df_to_file_params)

            end = time.perf_counter() - start
            logger.info(
                f"Job complete: took {end:0.2f} " 
                f"seconds for index {index} to {index + df.shape[0] - 1}"
            )

        save_task = asyncio.create_task(
            _save_with_log()
        )
        gcs_reqs.append(save_task)

        bq_api_calls.task_done()


@dataclass
class QueryParams:
    client: bigquery.client.Client
    index: int
    batch_size: int


async def perform_query(query_params: QueryParams):
    logger.info("Performing query...")

    query = make_query(query_params.index, query_params.batch_size)
    df = await bq_query(query_params.client, query)

    logger.info("Query fetched.")
    return df


def make_query(index: int, size: int) -> str:
    """Creates from base query and parameters"""
    logger.debug("Creating query...")

    return BASE_QUERY + f"\nLIMIT {size}\nOFFSET {index};"


async def bq_query(bq: bigquery.client.Client, query: str) -> pd.DataFrame:
    """Non-blocking BigQuery.query()"""
    logger.debug("Sending query...")

    loop = asyncio.get_event_loop()

    future = loop.create_future()

    def callback(job):
        result = job.result().to_dataframe()
        loop.call_soon_threadsafe(future.set_result, result)

    job = await loop.run_in_executor(None, bq.query, query)
    job.add_done_callback(callback)

    return await future


@dataclass
class EmbedParams:
    df: pd.DataFrame
    total_requests: int
    text_column: str                = TEXT_COLUMN
    embedding_encoding: str         = EMBEDDING_ENCODING
    embedding_model: str            = EMBEDDING_MODEL
    context_size: int               = EMBEDDING_CTX_LENGTH
    embedding_size: int             = EMBEDDING_SPACE
    requests_per_day_limit: int     = OPENAI_RPD_LIMIT


async def embed_text(
    df: pd.DataFrame, 
    total_requests: int, 
    embed_params: EmbedParams
) -> tuple[pd.DataFrame, int]:
    """Embeds the text column of dataframe, adds it at columns, and adds the 
    token count"""
    logger.info("Embedding text...")

    df["n_tokens"] = token_length_of_text(df, embed_params)

    df["group"] = chunk_ids_from_token_length(df["n_tokens"], embed_params)

    embeddings, total_requests = await embed_dataframe(
        df, 
        total_requests, 
        embed_params
    )

    df = df.drop([embed_params.text_column, 'group'], axis=1)

    df = pd.merge(df, embeddings, left_index=True, right_index=True)

    logger.info(f"Total requests to OpenAI so far: {total_requests}")

    return df, total_requests


def token_length_of_text(
    df: pd.DataFrame,
    embed_params: EmbedParams
) -> pd.Series:
    """Generate token length series from text series"""
    encoding = tiktoken.get_encoding(embed_params.embedding_encoding)

    return df[embed_params.text_column].apply(lambda x: len(encoding.encode(x)))


def chunk_ids_from_token_length(
    token_count_series: pd.Series,
    embed_params: EmbedParams
) -> pd.Series:
    """Generates a series that represents which group each text belongs to such
    that each group is under the context size limit"""

    group_identifiers = np.zeros(len(token_count_series))

    current_group_token_count = 0
    current_group = 0
    for index, token_count in enumerate(token_count_series):
        if current_group_token_count + token_count > embed_params.context_size:
            current_group += 1
            current_group_token_count = 0
        
        current_group_token_count += token_count
        group_identifiers[index] = current_group

    return pd.Series(group_identifiers, dtype=int)


async def embed_dataframe(
    df: pd.DataFrame, 
    total_requests: int,
    embed_params: EmbedParams
) -> tuple[pd.DataFrame, int]:
    """Splits dataframe into chunks that fit within the context token length"""
    # variables for logging data
    processed_batches: int = 0
    total_batches: int = 0
    total_time_for_embeddings: float = 0.
    total_length_so_far: int = 0

    # chunk text column into groups
    openai_queue: asyncio.Queue[pd.Series] = asyncio.Queue()
    grouped = df.groupby('group')
    for _, group in grouped:
        total_batches += 1
        openai_queue.put_nowait(group[embed_params.text_column])

    # list of results from open ai
    tasks: list[pd.DataFrame] = []

    remaining_chunks = lambda: not openai_queue.empty()
    remaining_requests = lambda: total_requests < embed_params.requests_per_day_limit

    while remaining_chunks() and remaining_requests():
        text_column_series = await openai_queue.get()

        start_time = time.perf_counter()

        try: 
            response = await async_wrap_get_embeddings(
                text_column_series,
                embed_params
            )
            total_requests += 1

            # printing status
            processed_batches += 1
            total_length_so_far += len(text_column_series)
            end_time = time.perf_counter() - start_time
            total_time_for_embeddings += end_time
            print_progress_bar(
                processed_batches, 
                total_batches, 
                total_length_so_far / total_time_for_embeddings
            )

            tasks.append(response)
        except ( Exception ) as e:
            logger.warn(
                "Failed to fetch embeddings with backoff, adding back to queue."
            )
            openai_queue.put_nowait(text_column_series)

    return pd.concat(tasks), total_requests
        

async def async_wrap_get_embeddings(
    text: pd.Series,
    embed_params: EmbedParams
) -> pd.DataFrame:
    """Fetch OpenAI embeddings."""
    logger.debug("Fetching OpenAI embeddings...")

    start_time = time.perf_counter()

    loop = asyncio.get_event_loop()

    embeddings_df = await loop.run_in_executor(
        None, 
        lambda args: get_embeddings(**args), 
        { 
            'text':         text, 
            'embed_params': embed_params
        }
    )

    end_time = time.perf_counter() - start_time

    logger.debug(f"Fetched {len(text)} embeddings in {end_time} seconds.")

    return embeddings_df


@retry(
    wait=wait_random_exponential(min=1, max=20), 
    stop=stop_after_attempt(6), 
    retry=retry_if_not_exception_type(openai.InvalidRequestError)
)
def get_embeddings(
    text: pd.Series, 
    embed_params: EmbedParams
) -> pd.DataFrame:
    """Get embeddings of a list of text/tokens with backoff from OpenAI"""
    embed_response = openai.Embedding.create(
        input=text.to_list(), 
        model=embed_params.embedding_model
    )["data"]

    embeddings_df = pd.DataFrame(
        np.nan, 
        index=range(len(text)), 
        columns=[f'dim_{i}' for i in range(embed_params.embedding_size)]
    )

    for obj in embed_response:
        index = obj['index']
        embedding = obj['embedding']
        embeddings_df.iloc[index] = embedding

    return embeddings_df


# https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
def print_progress_bar (
    processed_batches: int, 
    total: int, 
    avg_embeddings_per_sec: float
):
    """Call in a loop to create terminal progress bar"""
    bar_length = 50
    avg_embeddings = f"Avg embeddings/sec: {avg_embeddings_per_sec:0.2f}"
    percent = ("{0:.2f}").format(100 * (processed_batches / float(total)))
    filled_length = int(bar_length * processed_batches // total)
    bar = "█" * filled_length + '-' * (bar_length - filled_length)
    print(f"\rProgress: |{bar}| {percent}% : {avg_embeddings}", end = "\r")
    if processed_batches == total: 
        print()


@dataclass
class SaveDFToFileParams:
    df: pd.DataFrame
    filename: str
    bucket_name: str = BUCKET_NAME


async def save_df_to_file(save_df_to_file_params: SaveDFToFileParams) -> None:
    """Non-blocking call to save dataframe to file"""
    logger.info("Saving dataframe...")
    loop = asyncio.get_event_loop()

    file_obj = dataframe_to_bytesio(save_df_to_file_params.df)
    save_bytes_to_file_params = SaveBytesToFileParams(
        file_obj,
        save_df_to_file_params.filename
    )

    async def _upload():
        try:
            logger.debug("Uploading to GCS...")
            await loop.run_in_executor(
                None, 
                upload_to_gcs, 
                save_bytes_to_file_params
            )
        except Exception as e:
            logger.warning(f"Failed to upload to GCS. Saving locally...")
            await loop.run_in_executor(
                None, 
                save_locally,
                save_bytes_to_file_params
            )

    await _upload()


def dataframe_to_bytesio(df: pd.DataFrame) -> BytesIO:
    """Convert dataframe to bytes"""
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


@dataclass
class SaveBytesToFileParams:
    file_obj: BytesIO
    filename: str
    bucket_name: str = BUCKET_NAME


def upload_to_gcs(save_bytes_to_file_params: SaveBytesToFileParams) -> None:
    """Streams bytes of the dataframe to Google Cloud Storage"""
    start = time.perf_counter()

    storage_client = storage.Client()
    bucket = storage_client.bucket(save_bytes_to_file_params.bucket_name)
    blob = bucket.blob(save_bytes_to_file_params.filename)
    save_bytes_to_file_params.file_obj.seek(0)
    blob.upload_from_file(save_bytes_to_file_params.file_obj)

    end = time.perf_counter() - start
    logger.info(
        f"Uploaded {save_bytes_to_file_params.filename} "
        f"to GCS in {end:0.2f} seconds."
    )


def save_locally(save_bytes_to_file_params: SaveBytesToFileParams) -> None:
    """Save data locally"""
    with open(f"data/{save_bytes_to_file_params.filename}", 'wb') as f_local:
        save_bytes_to_file_params.file_obj.seek(0)
        f_local.write(save_bytes_to_file_params.file_obj.read())

    logger.info(
        f"Data saved locally at data/{save_bytes_to_file_params.filename}."
    )


if __name__ == "__main__":
    """CLI argument settings"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q", 
        "--query", 
        type=str, 
        default="./query.sql", 
        help="File path to the base SQL query."
    )
    parser.add_argument(
        "-b", 
        "--batch", 
        type=int, 
        default=ARG_DEFAULT_BATCH, 
        help="Batch sizes to pull from BigQuery"
    )
    parser.add_argument(
        "-s", 
        "--size", 
        type=int, 
        default=ARG_DEFAULT_SIZE, 
        help="Size of the BigQuery dataset"
    )
    parser.add_argument(
        "-i", 
        "--index", 
        type=int, 
        default=ARG_DEFAULT_INDEX, 
        help="Starting index in the BigQuery dataset"
    )
    parser.add_argument(
        "-v", 
        "--verbose", 
        action='store_true', 
        help="Flag to make logger more verbose"
    )
    parser.add_argument(
        "-r", 
        "--requests", 
        type=int, 
        default=0, 
        help="Number of requests to OpenAI already used for today"
    )

    ns = parser.parse_args()

    configure_logger(logger, ns.verbose)

    start = time.perf_counter()

    asyncio.run(main(
        ns.index,
        ns.batch,
        ns.size,
        ns.requests
    ))

    end = time.perf_counter() - start
    print(f"Program finished in {end:0.2f} seconds.")

