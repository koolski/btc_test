"""
Statistical Arbitrage Strategy Backtest for two exchanges.
"""

import pandas as pd
from typing import Optional
from strategy.utils import OrderBookDF
from strategy.utils import ExchangeName
from strategy.utils import BidAskSide
from logging import getLogger

logger = getLogger(__name__)

TIMESTAMP_TOLERANCE_MS = 10


class StatisticalArbitrageBacktest:
    """
    Statistical Arbitrage strategy class.

    The strategy implemented here looks for arbitrage opportunities between two exchanges
    by comparing the spread with its rolling mean and standard deviation.
    If it happens that the spread is more than 2 standard deviations above the mean,
    the strategy will buy on the exchange with the lower price and sell on the exchange
    with the higher price. If the spread is more than 2 standard deviations below the mean,
    the strategy will buy on the exchange with the higher price and sell on the exchange
    with the lower price.
    """

    def __init__(
        self, max_position_usd: int, fees: dict[ExchangeName, float], latency_ms: float
    ):
        """
        Initialize the Statistical Arbitrage strategy class.
        """
        self.max_position_usd = max_position_usd
        self.fees = fees
        self.latency_ms = latency_ms
        self.unified_orderbook = Optional[pd.DataFrame]
        self.pnl_over_time: list = []
        self.timestamps: list = []
        self.total_profit = 0
        self.total_volume = 0
        self.exchange_1_position = 0
        self.exchange_2_position = 0

    def set_data(self, orderbook_dict: dict[ExchangeName, OrderBookDF]) -> None:
        """
        Set and align historical order book data from multiple exchanges.
        """
        if len(orderbook_dict) != 2:
            raise ValueError("Provide order book data for exactly two exchanges.")

        logger.info(f"Setting orderbook data for {', '.join(self.exchange_names)}...")

        processed_orderbooks = {
            exchange: (
                orderbook.assign(
                    timestamp_ms=(orderbook["timestamp"].astype(int) // 10**6)
                ).sort_values("timestamp_ms")
            )
            for exchange, orderbook in orderbook_dict.items()
        }

        exchanges = self.exchange_names
        self.unified_orderbook = (
            pd.merge_asof(
                processed_orderbooks[exchanges[0]],
                processed_orderbooks[exchanges[1]],
                on="timestamp_ms",
                by="side",
                tolerance=TIMESTAMP_TOLERANCE_MS,
                direction="nearest",
                suffixes=(f"_{exchanges[0]}", f"_{exchanges[1]}"),
            )
            .dropna(subset=[f"timestamp_{exchanges[0]}"])
            .copy()
        )

        logger.info("Order book data successfully set")

    def get_data(self) -> pd.DataFrame:
        """
        Get the unified order book data.
        """
        return self.unified_orderbook

    def execute_trade(
        self,
        side: BidAskSide,
        expected_side: BidAskSide,
        trade_volume: float,
        price_1: float,
        price_2: float,
        exchange_1: ExchangeName,
        exchange_2: ExchangeName,
    ) -> tuple[float, float]:
        """
        Execute a trade and calculate profits for both exchanges.
        """
        if side == expected_side:
            profit_1 = (trade_volume / price_1) * (1 - self.fees[exchange_1])
            profit_2 = -(trade_volume / price_2) * (1 - self.fees[exchange_2])
        else:
            profit_2 = (trade_volume / price_2) * (1 - self.fees[exchange_2])
            profit_1 = -(trade_volume / price_1) * (1 - self.fees[exchange_1])

        return profit_1, profit_2

    @property
    def exchange_names(self) -> list[ExchangeName]:
        """
        Get the exchange names used in the arbitrage strategy.
        """
        return list(self.fees.keys())

    def close_positions(self, latency_adjusted_df: pd.DataFrame) -> None:
        """
        Close remaining open positions at the end of the trading session.
        """
        exchange_1, exchange_2 = self.exchange_names

        def get_last_price(df, side, column_name):
            side_df = df[df["side"] == side]
            return side_df[column_name].iloc[-1]

        if self.exchange_1_position != 0:
            last_bid_price_1 = get_last_price(
                latency_adjusted_df, BidAskSide.bid, f"price_{exchange_1}_latency"
            )
            last_ask_price_1 = get_last_price(
                latency_adjusted_df, BidAskSide.ask, f"price_{exchange_1}_latency"
            )

            if self.exchange_1_position > 0:
                # close long position: sell at bid price
                close_price = last_bid_price_1
                close_profit = -(self.exchange_1_position / close_price) * (
                    1 - self.fees[exchange_1]
                )
            else:
                # close short position: buy at ask price
                close_price = last_ask_price_1
                close_profit = (abs(self.exchange_1_position) / close_price) * (
                    1 - self.fees[exchange_1]
                )

            self.total_profit += close_profit
            self.total_volume += abs(self.exchange_1_position)

            logger.info(
                f"Closed position on {exchange_1}: Profit = {close_profit:.8f} BTC"
            )
            self.exchange_1_position = 0

        if self.exchange_2_position != 0:
            last_bid_price_2 = get_last_price(
                latency_adjusted_df, BidAskSide.bid, f"price_{exchange_2}_latency"
            )
            last_ask_price_2 = get_last_price(
                latency_adjusted_df, BidAskSide.ask, f"price_{exchange_2}_latency"
            )

            if self.exchange_2_position > 0:
                # close long position: sell at bid price
                close_price = last_bid_price_2
                close_profit = -(self.exchange_2_position / close_price) * (
                    1 - self.fees[exchange_2]
                )
            else:
                # close short position: buy at ask price
                close_price = last_ask_price_2
                close_profit = (abs(self.exchange_2_position) / close_price) * (
                    1 - self.fees[exchange_2]
                )

            self.total_profit += close_profit
            self.total_volume += abs(self.exchange_2_position)

            logger.info(
                f"Closed position on {exchange_2}: Profit = {close_profit:.8f} BTC"
            )
            self.exchange_2_position = 0

    def run(self, batch_size: int = 1_000_000) -> tuple[float, float]:
        """
        Run the statistical arbitrage strategy in batches to reduce memory usage.
        """
        self.total_profit = 0
        self.total_volume = 0

        self.exchange_1_position = 0
        self.exchange_2_position = 0

        exchange_1, exchange_2 = self.exchange_names

        logger.info(
            f"Running Statistical Arbitrage strategy for exchanges {exchange_1} and {exchange_2}..."
        )

        # Add latency-adjusted timestamp
        self.unified_orderbook["effective_timestamp_ms"] = (
            self.unified_orderbook["timestamp_ms"] + self.latency_ms
        )

        latency_adjusted_df = pd.merge_asof(
                self.unified_orderbook,
                self.unified_orderbook,
                left_on="effective_timestamp_ms",
                right_on="timestamp_ms",
                suffixes=("", "_latency"),
                direction="backward",
            ).dropna(
                subset=[f"price_{exchange_1}_latency", f"price_{exchange_2}_latency"]
            )

        # Calculate price spread
        latency_adjusted_df["spread"] = latency_adjusted_df.apply(
            lambda row: (
                row[f"price_{exchange_2}_latency"]
                - row[f"price_{exchange_1}_latency"]
                if row["side"] == BidAskSide.bid
                else row[f"price_{exchange_1}_latency"]
                     - row[f"price_{exchange_2}_latency"]
            ),
            axis=1,
        )

        # Calculate spread mean and std based on rolling window
        latency_adjusted_df["spread_mean"] = (
            latency_adjusted_df["spread"].rolling(window=10).mean()
        )
        latency_adjusted_df["spread_std"] = (
            latency_adjusted_df["spread"].rolling(window=10).std()
        )

        num_batches = (len(latency_adjusted_df) + batch_size - 1) // batch_size
        logger.info(f"Processing orderbook data in {num_batches} batches...")

        for batch_start in range(0, len(latency_adjusted_df), batch_size):
            batch = latency_adjusted_df.iloc[batch_start: batch_start + batch_size]

            for _, row in batch.iterrows():
                if pd.isna(row["spread_mean"]) or pd.isna(row["spread_std"]):
                    continue

                side = row["side"]

                max_quantity_exchange_1 = (
                    self.max_position_usd / row[f"price_{exchange_1}_latency"]
                )
                max_quantity_exchange_2 = (
                    self.max_position_usd / row[f"price_{exchange_2}_latency"]
                )

                remaining_capacity_exchange_1 = max(
                    0, max_quantity_exchange_1 - abs(self.exchange_1_position)
                )
                remaining_capacity_exchange_2 = max(
                    0, max_quantity_exchange_2 - abs(self.exchange_2_position)
                )

                # Calculate trade volume based on constraints
                trade_volume = min(
                    row[f"volume_{exchange_1}_latency"],
                    row[f"volume_{exchange_2}_latency"],
                    remaining_capacity_exchange_1,
                    remaining_capacity_exchange_2,
                )

                # Entry conditions for arbitrage
                if row["spread"] > row["spread_mean"] + 2 * row["spread_std"]:
                    exchange_1_profit, exchange_2_profit = self.execute_trade(
                        side,
                        BidAskSide.bid,
                        trade_volume,
                        row[f"price_{exchange_1}_latency"],
                        row[f"price_{exchange_2}_latency"],
                        exchange_1,
                        exchange_2,
                    )
                    self.total_profit += exchange_1_profit + exchange_2_profit
                    self.total_volume += trade_volume
                    self.exchange_1_position += (
                        trade_volume if side == BidAskSide.bid else -trade_volume
                    )
                    self.exchange_2_position -= (
                        trade_volume if side == BidAskSide.bid else trade_volume
                    )

                elif row["spread"] < row["spread_mean"] - 2 * row["spread_std"]:
                    exchange_1_profit, exchange_2_profit = self.execute_trade(
                        side,
                        BidAskSide.ask,
                        trade_volume,
                        row[f"price_{exchange_1}_latency"],
                        row[f"price_{exchange_2}_latency"],
                        exchange_1,
                        exchange_2,
                    )
                    self.total_profit += exchange_1_profit + exchange_2_profit
                    self.total_volume += trade_volume
                    self.exchange_2_position += (
                        trade_volume if side == BidAskSide.bid else -trade_volume
                    )
                    self.exchange_1_position -= (
                        trade_volume if side == BidAskSide.bid else trade_volume
                    )

            logger.info(
                f"Processed batch {batch_start // batch_size + 1}/{num_batches}, "
                f"Profit: {self.total_profit:.8f} BTC"
            )

        self.close_positions(latency_adjusted_df)

        return self.total_profit, self.total_volume
