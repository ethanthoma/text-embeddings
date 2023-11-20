from __future__ import annotations

from warnings import filterwarnings
filterwarnings(
    "ignore", 
    "Your application has authenticated using end user credentials"
)

from pymonad.tools import curry
from pymonad.either import Either, Left, Right

from dataclasses import dataclass
from google.cloud import bigquery as bq # type: ignore
from pathlib import Path
from pandas import DataFrame

from lib.tuple_mapper import tpm


@dataclass
class FetchParams:
    client: bq.Client
    cache_dir: str      = "./cache"


@dataclass
class FetchOutput:
    filename: str

    def __iter__(self) -> Iterator[Any]:
        return iter(dataclasses.astuple(self))


class FetchProcess:
    def __init__(self, params: FetchParams) -> None:
        self.params = params
        self.create_cache_folder()


    def create_cache_folder(self):
        Path(self.params.cache_dir).mkdir(parents=True, exist_ok=True)


    def __call__(
        self,
        index:                  int,
        size_of_current_batch:  int,
        query:                  str
        ) -> Either[str, FetchOutput]:

        return (
            self.perform_query(query)
            .then(self.check_dataframe_size)
            .then(self.generate_filename(index))
            .then(self.save_dataframe_to_file)
            .then(self.create_output)
        )

    
    def perform_query(self, sql_query: str) -> Either[str, DataFrame]:
        print(f"{self.__class__.__name__}: querying...")
        # DELETE ME
        return Right(DataFrame({'review_id': ['james', 'billy'], 'text': ['value one', 'value two']}))
        try: 
            client  = self.params.client
            job     = client.query(sql_query)
            df      = job.result().to_dataframe()
            return Right(df)
        except Exception as e:
            return Left(f"{self.__class__.__name__}: failed to perform query to client as {e}.")


    def check_dataframe_size(
        self, 
        df: DataFrame
        ) -> Either[str, DataFrame]:
        print(f"{self.__class__.__name__}: checking size...")
        if len(df.index) == 0:
            return Left(f"{self.__class__.__name__}: query result empty.")
        return Right(df)


    @curry(3)
    def generate_filename(
        self, 
        index:  int, 
        df:     DataFrame
        ) -> Right[Tuple[DataFrame, str]]:
        filename = f"raw-{index}_{index + df.shape[0]}.csv"
        return Right((df, filename))


    @tpm
    def save_dataframe_to_file(
        self, 
        df: pd.DataFrame, 
        filename: str
        ) -> Either[str, str]:
        save_path = f"{self.params.cache_dir}/{filename}"

        print(f"{self.__class__.__name__}: saving to {save_path}...")

        try:
            df.to_csv(save_path, index=False)
            return Right(filename)
        except Exception as e:
            return Left(f"{self.__class__.__name__}: failed to save dataframe as {e}.")
        

    def create_output(self, filename: str) -> Right[FetchOutput]:
        return Right(FetchOutput(filename))

