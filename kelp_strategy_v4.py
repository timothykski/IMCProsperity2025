import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class StrategyConfig:
    def __init__(self):
        # We'll dynamically set theta from the maximum drift
        self.theta = None

        # Example drift-based sizing (currently not strictly used, but a placeholder for expansions)
        self.drift_size_map = [
            (0.05, 0.2, (5, 10)),
            (0.2, 0.5, (10, 15)),
            (0.5, float("inf"), (15, 20))
        ]

        # Passive quoting thresholds
        self.passive_size_spread_2 = (10, 15)
        self.passive_size_spread_3 = (5, 10)

        # Per the problem statement:
        # max position ±50 from limit orders.
        # Market orders do not “count” in the environment’s limit,
        # but *you still have a net position* that can’t exceed ±50 (safe to check).
        self.max_position = 50

        # If we see a large trade in the tape:
        self.large_trade_volume = 35

        # Timings
        self.signal_decay_ms = 200  # how often we can re-signal
        self.cooldown_ms = 100  # Strategy B cooldown
        # limit order cancellation not allowed, so no TTL in revised code

        # Execution sizes
        self.execution_unit_size = 1

        # Passive fill probability logic
        self.min_fill_probability = 0.2

        # OBI threshold for Strategy A
        self.obi_threshold = 0.01


# Data structure
class LimitOrder:
    """Represent a resting limit order in the order book, uncancelable."""

    def __init__(self, ts, side, size, limit_price):
        self.ts = ts
        self.side = side  # +1 = buy, -1 = sell
        self.size = size
        self.limit_price = limit_price
        self.is_filled = False
        self.fill_ts = None
        self.fill_price = None


class ExecutionRecord:
    """Track each fill event for PnL analysis."""

    def __init__(self, timestamp, ex_type, side, size, price):
        self.timestamp = timestamp
        self.type = ex_type  # "market_order", "limit_fill", "impact_order", etc.
        self.side = side
        self.size = size
        self.price = price


# Trading strategy
class TradeKELP:
    def __init__(self, prices_df: pd.DataFrame, trades_df: pd.DataFrame, config: StrategyConfig):
        self.prices = prices_df.copy().reset_index(drop=True)
        self.trades = trades_df.copy().reset_index(drop=True)
        self.config = config

        self.execution_log = []  # records of all executions
        self.limit_orders = []   # uncancelable limit orders
        self.net_position = 0

        # Track when we entered a position (for time-based exit)
        self.position_entry_ts = None

        self.last_execution_ts = -np.inf
        self.last_limit_ts = -np.inf

    def run(self):
        self.preprocess_data()

        row_count = 0
        for idx, row in self.prices.iterrows():
            row_count += 1
            ts = row["timestamp"]
            mid = row["mid_price"]
            spread = row["spread"]
            drift = row["expected_drift"]
            obi_smooth = row["obi_smooth"]
            tfi = row["tfi"]

            # 1) Fill limit orders if the market crosses their price
            self.fill_limit_orders(ts, mid)

            # 2) Possibly do a time-based exit if we've held too long
            self.check_time_exit(row)

            # 3) If within signal cooldown, skip
            if ts - self.last_execution_ts < self.config.signal_decay_ms:
                continue

            # Strategy A
            if (
                spread <= 2
                and abs(drift) > self.config.theta
                and abs(obi_smooth) > self.config.obi_threshold
                and np.sign(obi_smooth) == np.sign(tfi)
            ):
                side = int(np.sign(drift))
                # check position limit
                if abs(self.net_position + side * self.config.execution_unit_size) <= self.config.max_position:
                    # fill price is top of book
                    fill_price = row["ask_price_1"] if side > 0 else row["bid_price_1"]
                    self.execute_trade(ts, "market_order", side, self.config.execution_unit_size, fill_price)
                    self.last_execution_ts = ts

            # STRATEGY B: Passive Limit
            elif (
                0.005 <= abs(obi_smooth) <= 0.01
                and (np.sign(obi_smooth) == np.sign(tfi) or abs(tfi) < 1e-6)
                and spread in [2, 3]
                and (ts - self.last_limit_ts >= self.config.cooldown_ms)
            ):
                fill_prob = self.estimate_fill_probability(spread, obi_smooth)
                if fill_prob >= self.config.min_fill_probability:
                    side = 1 if obi_smooth > 0 else -1
                    if abs(self.net_position + side * self.config.execution_unit_size) <= self.config.max_position:
                        # place limit near the best quote
                        if spread == 3:
                            # more aggressive offset
                            limit_price = row["ask_price_1"] if side > 0 else row["bid_price_1"]
                        else:
                            # if spread=2, a little closer to mid
                            limit_price = (
                                row["ask_price_1"] - 0.5 if side > 0 else row["bid_price_1"] + 0.5
                            )
                        self.place_limit_order(ts, side, self.config.execution_unit_size, limit_price)
                        self.last_execution_ts = ts
                        self.last_limit_ts = ts

            # STRATEGY C: Impact Trigger
            recent_trade = self.trades[self.trades["timestamp"] <= ts].tail(1)
            if not recent_trade.empty:
                last_vol = recent_trade.iloc[0]["quantity"]
                if (
                    last_vol >= self.config.large_trade_volume
                    and np.sign(obi_smooth) == np.sign(tfi)
                    and abs(drift) > self.config.theta
                ):
                    side = int(np.sign(tfi))
                    if abs(self.net_position + side * self.config.execution_unit_size) <= self.config.max_position:
                        fill_price = row["ask_price_1"] if side > 0 else row["bid_price_1"]
                        self.execute_trade(ts, "impact_order", side, self.config.execution_unit_size, fill_price)
                        self.last_execution_ts = ts

            # 4) AFTER placing potential new trades, check for a signal flip exit
            self.check_signal_flip(row)

        # After loop: final attempt to fill leftover limit orders
        final_ts = self.prices["timestamp"].iloc[-1]
        final_mid = self.prices["mid_price"].iloc[-1]
        self.fill_limit_orders(final_ts + 1, final_mid)

        print(f"Strategy processed {row_count} price updates.")
        return self.execution_log

    # Helper function

    def check_signal_flip(self, row):
        """If we hold a position but the signal flips strongly, flatten immediately."""
        if self.net_position == 0:
            return
        drift = row["expected_drift"]
        side_signal = int(np.sign(drift)) if abs(drift) > self.config.theta else 0
        side_held = int(np.sign(self.net_position))

        # If we are long but drift is strongly negative => flatten.
        # If short but drift is strongly positive => flatten.
        if side_signal != 0 and side_signal != side_held and side_held != 0:
            # Flatten
            flatten_side = -side_held
            size_to_exit = abs(self.net_position)
            fill_price = row["ask_price_1"] if flatten_side > 0 else row["bid_price_1"]
            self.execute_trade(row["timestamp"], "signal_flip_exit", flatten_side, size_to_exit, fill_price)

    def check_time_exit(self, row):
        """
        If we've held a position for 2000ms and the signals are no longer supportive,
        flatten. You can adjust or remove if it's hurting PnL.
        """
        if self.net_position == 0 or self.position_entry_ts is None:
            return

        ts_now = row["timestamp"]
        if (ts_now - self.position_entry_ts) > 2000:
            # Check if drift sign still aligns
            drift = row["expected_drift"]
            side_signal = int(np.sign(drift)) if abs(drift) > self.config.theta else 0
            side_held = int(np.sign(self.net_position))

            # If they no longer match => flatten
            if side_signal != side_held:
                flatten_side = -side_held
                size_to_exit = abs(self.net_position)
                fill_price = row["ask_price_1"] if flatten_side > 0 else row["bid_price_1"]
                self.execute_trade(ts_now, "time_stop_exit", flatten_side, size_to_exit, fill_price)

    def place_limit_order(self, ts, side, size, limit_price):
        lo = LimitOrder(ts, side, size, limit_price)
        self.limit_orders.append(lo)
        print(f"[PASSIVE] Place limit side={side}, px={limit_price:.2f}, pos={self.net_position}")

    def fill_limit_orders(self, now_ts, now_mid):
        for lo in self.limit_orders:
            if lo.is_filled:
                continue
            # side>0 => fill if mid <= limit_price
            # side<0 => fill if mid >= limit_price
            if lo.side > 0 and now_mid <= lo.limit_price:
                lo.is_filled = True
                lo.fill_ts = now_ts
                lo.fill_price = lo.limit_price
                self.execute_trade(now_ts, "limit_fill", lo.side, lo.size, lo.fill_price)
            elif lo.side < 0 and now_mid >= lo.limit_price:
                lo.is_filled = True
                lo.fill_ts = now_ts
                lo.fill_price = lo.limit_price
                self.execute_trade(now_ts, "limit_fill", lo.side, lo.size, lo.fill_price)

    def execute_trade(self, ts, ex_type, side, size, fill_price):
        # log
        self.execution_log.append(vars(ExecutionRecord(ts, ex_type, side, size, fill_price)))
        old_pos = self.net_position
        self.net_position += side * size

        # set entry time if we just went from 0 -> non-0
        if old_pos == 0 and self.net_position != 0:
            self.position_entry_ts = ts
        elif self.net_position == 0:
            self.position_entry_ts = None

        print(f"[TRADE] {ex_type}, side={side}, px={fill_price:.2f}, size={size}, newPos={self.net_position}")


    def estimate_fill_probability(self, spread, obi_smooth):
        #todo: remove, not used
        if spread == 2:
            return 0.6 if abs(obi_smooth) > 0.005 else 0.3
        elif spread == 3:
            return 0.3 if abs(obi_smooth) > 0.005 else 0.1
        return 0

    def preprocess_data(self):
        self.compute_mid_price()
        self.compute_spread()
        self.compute_obi()
        self.compute_tfi()
        self.compute_expected_drift()

    def compute_mid_price(self):
        self.prices["mid_price"] = (self.prices["bid_price_1"] + self.prices["ask_price_1"]) / 2

    def compute_spread(self):
        self.prices["spread"] = self.prices["ask_price_1"] - self.prices["bid_price_1"]

    def compute_obi(self):
        bid_vols = sum(self.prices.get(f"bid_volume_{i}", 0).fillna(0) for i in range(1, 4))
        ask_vols = sum(self.prices.get(f"ask_volume_{i}", 0).fillna(0) for i in range(1, 4))
        self.prices["obi"] = (bid_vols - ask_vols) / (bid_vols + ask_vols + 1e-6)
        self.prices["obi_smooth"] = self.prices["obi"].rolling(5, min_periods=1).mean()

    def compute_tfi(self):
        self.trades["timestamp_bin"] = (self.trades["timestamp"] // 1000) * 1000
        self.trades["direction"] = np.where(
            self.trades["price"] > self.trades["price"].shift(1), 1,
            np.where(self.trades["price"] < self.trades["price"].shift(1), -1, 0)
        )
        self.trades["signed_qty"] = self.trades["direction"] * self.trades["quantity"]
        tfi = self.trades.groupby("timestamp_bin")["signed_qty"].sum().reset_index(name="tfi")
        self.prices["timestamp_bin"] = (self.prices["timestamp"] // 1000) * 1000
        self.prices = self.prices.merge(tfi, on="timestamp_bin", how="left").fillna({"tfi": 0})

    def compute_expected_drift(self):
        self.prices["future_mid"] = self.prices["mid_price"].shift(-10)
        self.prices["expected_drift"] = (self.prices["future_mid"] - self.prices["mid_price"]).fillna(0)
        self.set_dynamic_theta()

    def set_dynamic_theta(self):
        max_drift = self.prices["expected_drift"].abs().max()
        self.config.theta = 0.5 * max_drift if max_drift > 0 else 0.1
        print(f"Dynamic theta set to: {self.config.theta:.4f}")


###############################################################################
# PnLEvaluator: Marked-To-Market Approach
###############################################################################
class PnLEvaluator:
    """
    1) Rebuild a trade-by-trade ledger to know exactly how position changes over time.
    2) For each price timestamp, we track:
        - Current net position
        - Realized PnL from trades that reduce position
        - Unrealized PnL from net_position * (current_mid_price - average_cost)
    3) At the end, total PnL = realized + unrealized(last time).
    """

    def __init__(self, logs: list, prices_df: pd.DataFrame):
        self.execs = pd.DataFrame(logs)
        self.prices = prices_df.copy().reset_index(drop=True)

        # Arrays to accumulate PnL over time
        self.times = []
        self.realized_pnl = []
        self.unrealized_pnl = []
        self.cum_pnl = []

    def compute(self):
        if self.execs.empty:
            print("No trades executed. PnL = 0.")
            return pd.DataFrame()

        # Sort all trades by timestamp & day
        self.execs.sort_values("timestamp", inplace=True)
        self.prices.sort_values("timestamp", inplace=True)

        # For incremental PnL tracking
        current_position = 0
        avg_cost = 0.0  # Weighted average cost (for the position we hold)
        total_realized = 0.0

        price_idx = 0
        price_len = len(self.prices)

        trade_idx = 0
        n_trades = len(self.execs)

        trades_list = self.execs.to_dict('records')

        results_rows = []

        while price_idx < price_len:
            rowP = self.prices.iloc[price_idx]
            now_ts = rowP['timestamp']
            now_mid = rowP['mid_price']

            # Process any trades at or before now_ts
            while trade_idx < n_trades and trades_list[trade_idx]['timestamp'] <= now_ts:
                tr = trades_list[trade_idx]
                side = tr['side']
                size = tr['size']
                fill_px = tr['price']

                old_position = current_position
                old_value = current_position * avg_cost

                # Update position
                new_position = current_position + side * size

                # Realized PnL occurs when you reduce or flip position
                if np.sign(new_position) == np.sign(old_position):
                    # same side => no immediate realization, just average cost adjustment
                    if old_position == 0:
                        avg_cost = fill_px
                    else:
                        # Weighted average cost
                        new_value = old_value + side * size * fill_px
                        if new_position != 0:
                            avg_cost = new_value / new_position
                        else:
                            avg_cost = 0.0
                    current_position = new_position
                else:
                    # position is reducing or flipping
                    # Example: old_position=10, side=-1, size=3 => realize PnL on 3
                    position_change = side * size
                    # Realized on partial
                    # let's break down carefully:
                    if abs(new_position) < abs(old_position):
                        # partial flatten
                        realized_part = position_change * (fill_px - avg_cost)
                        total_realized += realized_part
                        current_position = new_position
                    else:
                        # flipping or fully flatten
                        realized_part = -old_position * (fill_px - avg_cost)
                        total_realized += realized_part
                        current_position = new_position
                        leftover_size = abs(new_position)
                        if leftover_size > 0:
                            # The leftover “entry price” is basically this fill_px
                            avg_cost = fill_px
                        else:
                            avg_cost = 0.0

                trade_idx += 1

            # compute mark-to-market PnL (unrealized) + realized
            # Unrealized = position * (now_mid - avg_cost)
            # total PnL = realized + unrealized
            ur = current_position * (now_mid - avg_cost)
            cum = total_realized + ur

            self.times.append(now_ts)
            self.realized_pnl.append(total_realized)
            self.unrealized_pnl.append(ur)
            self.cum_pnl.append(cum)

            results_rows.append({
                "timestamp": now_ts,
                "position": current_position,
                "avg_cost": avg_cost,
                "realized_pnl": total_realized,
                "unrealized_pnl": ur,
                "total_pnl": cum
            })

            price_idx += 1

        results_df = pd.DataFrame(results_rows)
        return results_df

    def plot_cumulative_pnl(self):
        if len(self.times) == 0:
            print("No PnL to plot.")
            return

        plt.figure(figsize=(12, 6))

        # Plot total cumulative PnL
        plt.plot(
            self.times,
            self.cum_pnl,
            label='Total Cumulative PnL',
            linewidth=1.5,
            color='blue'
        )

        # Plot realized PnL (dashed line)
        plt.plot(
            self.times,
            self.realized_pnl,
            label='Realized PnL',
            linestyle='--',
            linewidth=1.5,
            color='orange'
        )

        plt.axhline(0, color='gray', linestyle='--')
        plt.xlabel("Timestamp (from all 3 days)")
        plt.ylabel("PnL")
        plt.title("Mark-to-Market & Realized PnL (3 Days Combined)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# Main
if __name__ == "__main__":
    prices = pd.read_excel("C:/Users/aa04947/OneDrive - APG/Desktop/IMCR1/out/kelp_prices_combined.xlsx")
    trades = pd.read_excel("C:/Users/aa04947/OneDrive - APG/Desktop/IMCR1/out/kelp_trades_combined.xlsx")


    cfg = StrategyConfig()
    kelp = TradeKELP(prices, trades, cfg)
    logs = kelp.run()

    if not logs:
        print("No trades executed.")
    else:
        evaluator = PnLEvaluator(logs, kelp.prices)
        results_df = evaluator.compute()
        evaluator.plot_cumulative_pnl()
        print(results_df.head(20))
