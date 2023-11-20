from __future__ import annotations
from typing import Tuple, Optional

from pymonad.either import Either, Left, Right
from pymonad.tools import curry

from math import ceil
from dataclasses import dataclass

@dataclass
class QueryParams:
    """
    Data class for holding query parameters.

    Attributes:
        batch_size  (int): The size of each batch.
        size        (int): Total size of items to process.
        index       (int): Starting index for processing.
        file_path   (str): Path to the SQL file.
        primary_key (Optional[str]): Primary key used in SQL query.
    """
    batch_size:     int
    size:           int
    index:          int = 0
    file_path:      str = ""
    primary_key:    Optional[str] = None


@dataclass
class QueryOutput:
    """
    Data class for holding the output of a query process.

    Attributes:
        index                   (int): Current index in processing.
        size_of_current_batch   (int): Size of the current batch.
        query                   (str): The SQL query string.
    """
    index:                  int
    size_of_current_batch:  int
    query:                  str

    def __iter__(self) -> Iterator[Any]:
        return iter(dataclasses.astuple(self))


class QueryProcess:
    """
    Class for processing SQL queries in batches.

    Attributes:
        params (QueryParams): Parameters for query processing.
    """
    def __init__(self, params: QueryParams):
        self.params = params
        self.total_batches = ceil(
            (params.size - params.index) / params.batch_size
        )
        self.current_batch = 0


    def __call__(self) -> Either[str, str]:
        """
        Executes the query process for a given batch.

        Returns:
            Either: Right with QueryOutput or Left with error message.
        """
        if self.current_batch >= self.total_batches:
            return Left(f"{self.__class__.__name__}: end of batches.")

        current_index, size_of_current_batch = self.calculate_batch_details()

        self.current_batch += 1

        return (
            self.read_base_query_from_file() 
            .then(self.add_order_to_base)
            .then(self.get_next(current_index, size_of_current_batch)) 
            .then(self.create_query_output(
                current_index, 
                size_of_current_batch
            ))
        )


    def calculate_batch_details(self) -> Tuple[int, int]:
        """
        Calculates the details of the current batch.

        Returns:
            Tuple[int, int]: A tuple containing the current index and the size 
            of the current batch.
        """
        current_index = self.params.index + (
            self.current_batch * self.params.batch_size
        )
        size_of_current_batch = min(
            self.params.batch_size, 
            self.params.size - current_index
        )

        return current_index, size_of_current_batch 


    def read_base_query_from_file(self) -> Either[str, str]:
        """
        Reads the base SQL query from a file.

        Returns:
            Either[str, str]: Right containing the query string if successful, 
            Left with error message otherwise.
        """
        try:
            with open(self.params.file_path, 'r') as f:
                base_query = f.read()
            return Right(base_query)
        
        except FileNotFoundError:
            return Left(f"{self.__class__.__name__}: failed to read from `{file_path}`.")

    
    def add_order_to_base(self, query: str) -> Either[str, str]:
        """
        Adds an ORDER BY clause to the base query if not present.

        Args:
            query (str): The base SQL query.

        Returns:
            Either[str, str]: Right with modified query or Left with error 
            message.
        """
        if not "ORDER BY" in query.upper():
            if not self.params.primary_key:
                return Left("Query Process: no ORDER BY in query or \
                            primary_key in params")
            query += f"\nORDER BY {self.params.primary_key}"
        return Right(query)


    @curry(4)
    def get_next(self, index: int, size: int, query: str) -> Right[str]:
        """
        Appends LIMIT and OFFSET clauses to the SQL query.

        Args:
            index   (int): Current index in processing.
            size    (int): Size of the current batch.
            query   (str): The SQL query.

        Returns:
            Either: Right with modified query.
        """
        query += f"\nLIMIT {size}\nOFFSET {index};"
        return Right(query)


    @curry(4)
    def create_query_output(
        self, 
        index:  int, 
        size:   int, 
        query:  str
        ) -> Right[str]:
        """
        Creates a QueryOutput object.

        Args:
            index (int): Current index in processing.
            size (int): Size of the current batch.
            query (str): The SQL query.

        Returns:
            Right[str]: Right with QueryOutput object.
        """
        return Right(QueryOutput(index, size, query))


    def __iter__(self) -> QueryProcess:
        """
        Initialize the iterator.

        Returns:
            QueryProcess: Returns self to be used as an iterator.
        """
        self.current_batch = 0
        return self


    def __next__(self) -> Either[str, str]:
        """
        Proceed to the next batch.

        Returns:
            Either: Right with QueryOutput or Left with error message.

        Raises:
            StopIteration: When all batches are processed.
        """
        if self.current_batch < self.total_batches:
            return self()
        else:
            raise StopIteration

