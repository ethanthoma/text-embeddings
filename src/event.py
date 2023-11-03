from enum import Enum


class Message(Enum):
    QUERY_EXIT = "QueryServiceExit"
    FETCH_EXIT = "FetchServiceExit"
    EMBED_EXIT = "EmbedServiceExit"
    STORE_EXIT = "StoreServiceExit"

