# Non-blocking OpenAI embedding via Google Cloud

This is code that to fetch OpenAI embeddings async.
It does three things:
- Perform a SQL query to Google BigQuery to fetch a batch size of data
- Chunk the batch within token limits and submit batch requests to OpenAI
- Merge OpenAI responses for the batch and upload to Google Cloud Storage

All blocking functions are wrapped in aioasync wrappers.

It performs both exponential backoff and requeue of failed embedding requests
until the batch is finished. Due to GPT's request limits, multiple threads 
wouldn't make a difference as most requests are around every 0.4 to 0.7 seconds.

I use Google Cloud AIO Storage library for uploading. As far as I could tell, I
could not get this to work async. so it is currently a sync call. I could not 
stream it either... :(

## 1: Building

You can build the code by running the following commands:
- `nix develop -i` in the root dir of the project
- `poetry install`

This "module" relies on `python=">=3.9,<3.13"`.

## 2: Running

To run the code, you will need to do two things:
- set required environment varibales of `POETRY_OPENAI_API_KEY` in a `.env`
- update the function `make_query`

Afterwards, you can simply run through nix and poetry:
- `nix deploy -i`
- `poetry run python src`

