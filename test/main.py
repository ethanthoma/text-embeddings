from multiprocessing import Manager


from services.query import QueryParams, QueryProcess
from services.fetch import FetchParams, FetchProcess
from lib.pipeline import Pipeline


# for test running the new code
with Manager() as manager:
    batch_size = 5
    size = 20
    index = 8
    filename = "query.sql"
    key = "review_id"

    batch = 0

    qp = QueryParams(batch_size, size, index, filename, key)
    qs = QueryProcess(qp)

    fetch = FetchProcess(FetchParams(None))

    pl = Pipeline(
        qs,
        fetch
    )

    pl.start()
    pl.join()

