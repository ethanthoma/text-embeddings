from __future__ import annotations

import argparse
from google.cloud import bigquery as bq
import multiprocessing as mp
import os
import time
from typing import Any

# local
from service import BaseService
from message import MessageBus
from query import QueryService, QueryParams
from fetch import FetchService, FetchParams
from embed import EmbedService, EmbedParams
from store import StoreService, StoreParams


ARG_DEFAULT_BATCH       = 10
ARG_DEFAULT_BUCKET      = "temp_embeddings"
ARG_DEFAULT_INDEX       = 0
ARG_DEFAULT_REQUESTS    = 0
ARG_DEFAULT_SIZE        = 20
ARG_DEFAULT_WORKERS     = 5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", 
        "--batch", 
        type=int, 
        default=ARG_DEFAULT_BATCH, 
        help="Batch sizes to pull from BigQuery"
    )
    parser.add_argument(
        "-B", 
        "--bucket", 
        type=str, 
        default=ARG_DEFAULT_BUCKET,
        help="Name of GCS bucket"
    )
    parser.add_argument(
        "-i", 
        "--index", 
        type=int, 
        default=ARG_DEFAULT_INDEX, 
        help="Starting index in the BigQuery dataset"
    )
    parser.add_argument(
        "-r", 
        "--requests", 
        type=int, 
        default=ARG_DEFAULT_REQUESTS, 
        help="Number of requests to OpenAI already used for today"
    )
    parser.add_argument(
        "-s", 
        "--size", 
        type=int, 
        default=ARG_DEFAULT_SIZE, 
        help="Size of the BigQuery dataset"
    )
    parser.add_argument(
        "-w", 
        "--workers", 
        type=int, 
        default=ARG_DEFAULT_WORKERS,
        help="Number of worker threads for calling OpenAI API"
    )

    ns = parser.parse_args()

    start = time.perf_counter()

    with mp.Manager() as manager:
        message_bus = MessageBus()

        # service args
        sql_query_queue: mp.Queue[tuple[int, int, str]] = mp.Queue()
        raw_filename_queue: mp.Queue[str] = mp.Queue()
        embed_filename_queue: mp.Queue[str] = mp.Queue()

        # define services
        services: list[tuple[BaseService, tuple[mp.Queue[Any], ...]]] = []

        query_params = QueryParams(batch_size=ns.batch, size=ns.size)
        services.append((
            QueryService(query_params),
            (sql_query_queue, )
        ))

        fetch_params = FetchParams(client=bq.Client())
        services.append((
            FetchService(fetch_params),
            (sql_query_queue, raw_filename_queue)
        ))

        embed_params = EmbedParams(
            openai_key=os.environ["POETRY_OPENAI_API_KEY"], 
            requests_so_far=ns.requests,
            max_workers = ns.workers
        )
        services.append((
            EmbedService(embed_params),
            (raw_filename_queue, embed_filename_queue)
        ))

        store_params = StoreParams(bucket_name=ns.bucket)
        services.append((
            StoreService(store_params),
            (embed_filename_queue, )
        ))

        processes: list[mp.Process] = [ ]
        for service, args in services:
            message_bus.register(service)
            processes.append(mp.Process(
                target=service.run,
                args=args,
                daemon=True
            ))

        # start services
        message_bus_process = mp.Process(target=message_bus.run, daemon=True)
        message_bus_process.start()

        for process in processes:
            process.start()

        # wait for services
        for process in processes:
            process.join()

        message_bus.stop()
        message_bus_process.terminate()

    end = time.perf_counter() - start
    print(f"Program finished in {end:0.2f} seconds.")

