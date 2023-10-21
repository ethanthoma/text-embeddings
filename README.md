# Non-blocking OpenAI embedding via Google Cloud

It does three things:
- Performs an SQL query to Google BigQuery to fetch a batch size of data
- Chunk the batch within token limits and submit chunked requests to OpenAI
- Merge chunk responses back into a batch and upload to Google Cloud Storage

All blocking functions are wrapped in aioasync wrappers.

The embedding calls to OpenAI use both exponential backoff and a requeue for any
failed requests that still failed after the backoff. There is a parameter to set
your daily limit of requests to OpenAI. After hitting the limit, it will stop 
making requests and save the current progress to storage.

**NOTE: since it retries your request until it your limit, it is possible for a
bad request to repeat and use all of your daily limit. Make sure your text 
column in your query matches the text column set by the constant.**

Due to GPT's request limits, multiple threads wouldn't make a difference as most 
requests are around every 0.4 to 0.7 seconds which is faster than the 1 second
threshold. This may change in the future or if you are lucky enough to get more
RPM.

It streams the batched data back into Google Cloud Storage. If it fails for any 
reason, it gets saved locally.

## 2: Running

To run the code, you will need to do two things:
- set the required environment variable of `POETRY_OPENAI_API_KEY` in a `.env` 
file in the root so poetry can read it into the environment
- update the function `make_query` to perform the query you want to perform

Afterwards, you can run the program through nix and poetry:
- `nix develop -i` in the root dir of the project
- `poetry run python src`
- `poetry install`

This "module" relies on `python=">=3.9,<3.13"`.

## 3: Guide

There is a help command available via `-h` or `--help`. The full list of 
commands are below:

| short | long       | action                                               |
|-------|------------|------------------------------------------------------|
| -b    | --batch    | Batch sizes to pull from BigQuery                    |
| -s    | --size     | Size of the BigQuery dataset                         |
| -i    | --index    | Starting index in the BigQuery dataset               |
| -v    | --verbose  | Flag to make logger more verbose                     |
| -r    | --requests | Number of requests to OpenAI already used for today  |
