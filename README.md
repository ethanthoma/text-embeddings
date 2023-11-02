# Multithreaded OpenAI embedding via Google Cloud

It does four things:
- Query: generate ordered SQL queries limited to a batch size
- Fetch: query GCS BigQuery and save locally
- Embed: chunk the fetched data and embed the text, saves locally
- Store: uploads to GCS Storage

## 1. Use

```
This "module" relies on python=">=3.9,<3.13".
````

I used poetry to manage my libraries, you will need to too. As it uses OpenAI
and GCS, you will have to authenticate for both. GCS should be done via the CLI
and OpenAI requires you to set the environment of `POETRY_OPENAI_API_KEY`. Then,
you can simply enter `poetry install` and then `poetry run python src`.

You will probably want to change the query performed. It is currently stored in
`src/query.py`. It **must** have an order attribute.

## 2: Guide

There is a help command available via `-h` or `--help`. The full list of 
commands are below:

| short | long       | action                                               |
|-------|------------|------------------------------------------------------|
| -b    | --batch    | Batch sizes to pull from BigQuery                    |
| -B    | --bucket   | Name of GCS bucket                                   |
| -i    | --index    | Starting index in the BigQuery dataset               |
| -s    | --size     | Size of the BigQuery dataset                         |
| -r    | --requests | Number of requests to OpenAI already used for today  |
| -w    | --workers  | Number of worker threads for calling OpenAI API      |

## 3: Deets

The embedding calls to OpenAI use both exponential backoff and multiple thread.
There is a parameter to set your request limits for OpenAI in `embed.py` or in 
the `__main__.py`. After hitting the limit, it will stop making requests and 
save the current progress to storage. The rest will be stored locally that you 
can choose to embed later.

**NOTE: I plan to autoload any locally stored chunks that were not embedded.**

