"""
Utility functions for processing order book data.
"""

from enum import StrEnum
from typing import NewType
from typing import TypedDict
from tempfile import NamedTemporaryFile

import dask.dataframe as dd
import pandas as pd
import requests
import gzip
import shutil

ExchangeName = NewType("ExchangeName", str)


class OrderBookDF(TypedDict):
    """
    Orderbook DataFrame type hint.
    """
    timestamp: pd.Timestamp
    side: str
    price: float
    volume: float


class BidAskSide(StrEnum):
    """
    Enum for bid and ask sides.
    """
    bid = "Bid"
    ask = "Ask"


def read_orderbook_from_url(url: str) -> pd.DataFrame:
    """
    Process compressed order book file given URL
    """
    with NamedTemporaryFile(delete=False, suffix=".txt.gz") as temp_gz_file:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Ensure the request was successful
        with open(temp_gz_file.name, "wb") as f_out:
            shutil.copyfileobj(response.raw, f_out)

    temp_txt_file = temp_gz_file.name.replace(".gz", "")
    with gzip.open(temp_gz_file.name, "rb") as f_in, open(temp_txt_file, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    timestamp_pattern = r"receive_time:\s([\d-]+\s[\d:.]+)"
    side_pattern = r"(Bid|Ask) price:"
    price_pattern = r"(?:Bid|Ask) price:\s([\d.]+)"
    volume_pattern = r"volume:\s([\d.]+)"

    raw_orderbook = dd.read_csv(temp_txt_file, header=None, names=["raw"])

    processed = raw_orderbook.assign(
        timestamp=raw_orderbook["raw"].str.extract(timestamp_pattern)[0],
        side=raw_orderbook["raw"].str.extract(side_pattern)[0],
        price=raw_orderbook["raw"].str.extract(price_pattern)[0].astype(float),
        volume=raw_orderbook["raw"].str.extract(volume_pattern)[0].astype(float),
    )

    processed["timestamp"] = processed["timestamp"].ffill()
    processed["timestamp"] = dd.to_datetime(
        processed["timestamp"], format="%y-%m-%d %H:%M:%S.%f"
    )

    processed = processed.dropna(subset=["side", "price", "volume"])
    processed = processed[["timestamp", "side", "price", "volume"]]

    return processed.compute()
