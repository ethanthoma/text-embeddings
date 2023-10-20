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


logging.basicConfig()
logger = logging.getLogger(__name__)


# argument defaults
ARG_DEFAULT_INDEX       = 0
ARG_DEFAULT_BATCH       = 10
ARG_DEFAULT_SIZE        = 20


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
    index: int, 
    batch: int, 
    size: int, 
    verbose: bool
) -> None:
    logger.setLevel(
        logging.DEBUG if verbose else logging.INFO
    )

    logger.debug("Initializating BigQuery API call tasks.")

    bq_api_calls = asyncio.Queue() # type: asyncio.Queue[int]
    for n in range(ceil(size / batch)):
        bq_api_calls.put_nowait((n * batch) + index)

    gcs_reqs = []

    logger.debug("Starting tasks...")
    asyncio.create_task(
        job(bq_api_calls, gcs_reqs, batch)
    )

    await bq_api_calls.join()

    await asyncio.gather(*gcs_reqs)


async def job(
    bq_api_calls: asyncio.Queue[int], 
    gcs_reqs: list,
    batch_size: int
) -> None:
    bq = bigquery.Client()
    
    while True:
        index = await bq_api_calls.get()
        logger.info(
            f"Job #{index} start: "
            f"with estimated size {batch_size}."
        )
        start = time.perf_counter()

        # fetch query data
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

        index += 1950000
        filename = f"batch-index_{index}_to_{index + df.shape[0] - 1}.csv"

        async def _save_with_log():
            await save_df_to_file(df, filename, BUCKET_NAME)

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


def make_query(index: int, size: int) -> str:
    """A query to fetch a batch of data"""
    logger.debug("Creating query...")

    return f"""
        SELECT r.review_id, r.text
        FROM Yelp_review_text.reviews r
        WHERE r.date > TIMESTAMP "2015-04-20 06:24:29 UTC"
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

    df = await future
    logger.debug("Query fetched.")
    return df


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
    
    openai_queue = asyncio.Queue() # type: asyncio.Queue[pd.Series]

    grouped = df.groupby('group')
    for _, group in grouped:
        openai_queue.put_nowait(group[text_column])

    tasks = [] # type: list[pd.DataFrame]
    while not openai_queue.empty():
        text_column_series = await openai_queue.get()

        logger.info("Attempting to fetch embeddings...")

        try: 
            response = await batch_get_embeddings_from_series_as_df(
                text_column_series,
                embedding_model,
                embedding_size
            )
            
            logger.info("Fetched embeddings.")

            tasks.append(response)
        except ( Exception ) as e:
            logger.warn(
                "Failed to fetch embeddings with backoff, adding back to queue."
            )
            openai_queue.put_nowait(text_column_series)

    embeddings = pd.concat(tasks)

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


async def save_df_to_file(
    df: pd.DataFrame,
    filename: str,
    bucket_name: str
) -> None:
    """Non-blocking call to save dataframe to file"""
    logger.info("Saving dataframe...")
    loop = asyncio.get_event_loop()

    file_obj = dataframe_to_bytesio(df)

    async def _upload():
        try:
            await loop.run_in_executor(
                None, 
                upload_to_gcs, 
                file_obj, 
                filename,
                bucket_name 
            )
        except Exception as e:
            logger.error(f"Failed to upload to GCS: {e}")
            await loop.run_in_executor(
                None,
                save_locally,
                file_obj,
                filename
            )

    await _upload()


def dataframe_to_bytesio(df: pd.DataFrame) -> BytesIO:
    """Convert dataframe to bytes"""
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


def upload_to_gcs(
    file_obj: BytesIO,
    filename: str,
    bucket_name: str
):
    """Streams bytes of the dataframe to Google Cloud Storage"""
    start = time.perf_counter()

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    file_obj.seek(0)
    blob.upload_from_file(file_obj)

    end = time.perf_counter() - start
    logger.info(
        f"Stream data uploaded to {filename} "
        f"in bucket {bucket_name} in {end:0.2f} seconds."
    )


def save_locally(
    file_obj: BytesIO, 
    filename: str 
):
    """Save data locally"""
    with open(filename, 'wb') as f_local:
        file_obj.seek(0)
        f_local.write(file_obj.read())

    logger.info(f"Data saved locally at {filename}")


if __name__ == "__main__":
    """CLI argument settings"""
    parser = argparse.ArgumentParser()
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


    ns = parser.parse_args()

    logger = logging.getLogger(__name__)

    start = time.perf_counter()

    asyncio.run(main(**ns.__dict__))

    end = time.perf_counter() - start
    print(f"Program finished in {end:0.2f} seconds.")

