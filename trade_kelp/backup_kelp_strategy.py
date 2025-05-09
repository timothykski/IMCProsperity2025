import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from order import LimitOrder, ExecutionRecord
from pnl_calculator import PnLEvaluator


# Trading strategy
class Trader:
    def __init__(self):
        """
        config must include (among others):
          - max_position: maximum allowed net active position (e.g., 50)
          - execution_unit_size: trade size per execution (e.g., 1)
          - signal_decay_ms: minimum milliseconds between trades
          - cooldown_ms: additional cooldown for passive orders
          - large_trade_volume: threshold volume for the impact trigger
          - obi_threshold: e.g., 0.01 for aggressive signals
          - min_fill_probability: e.g., 0.2 for passive orders
          - trailing_stop_distance: e.g., 2.0 ticks
          - breakout_threshold_up: e.g., 0.5 ticks for breakout up
          - breakout_threshold_down: e.g., 0.5 ticks for breakout down
          - daily_reentry_threshold: (new) threshold for reentry based on previous day’s close (e.g., 0.3 ticks)
        """
        self.prices = pd.read_excel("C:/Users/aa04947/OneDrive - APG/Desktop/IMCR1/out/kelp_prices_combined.xlsx")
        self.trades = pd.read_excel("C:/Users/aa04947/OneDrive - APG/Desktop/IMCR1/out/kelp_trades_combined.xlsx")
        self.config = {
            "theta": None,

            "drift_size_map": [
                (0.05, 0.2, (5, 10)),
                (0.2, 0.5, (10, 15)),
                (0.5, float("inf"), (15, 20))
            ],

            "passive_size_spread_2": (10, 15),
            "passive_size_spread_3": (5, 10),

            "max_position": 50,
            "large_trade_volume": 35,

            "signal_decay_ms": 200,
            "cooldown_ms": 100,

            "execution_unit_size": 1,
            "min_fill_probability": 0.2,

            "obi_threshold": 0.01,
            "trailing_stop_distance": 2.0,

            "breakout_threshold_up": 0.5,
            "breakout_threshold_down": 0.5,

            "daily_reentry_threshold": 0.3
        }

        # Execution & Orders
        self.execution_log = []
        self.limit_orders = []

        self.net_position = 0
        self.position_entry_ts = None  # Used for time-based exit

        self.last_execution_ts = -np.inf
        self.last_limit_ts = -np.inf

        # For trailing stop tracking
        self.best_price_in_position = None

    def run(self):
        """Main execution loop over each price update."""
        self.preprocess_data()

        row_count = 0
        current_day = None  # Track current day
        prev_day_close_mid = None  # Closing mid price from previous day

        # Iterate over prices, sorted by day then timestamp.
        for idx, row in self.prices.iterrows():
            row_count += 1
            this_day = row["day"]
            ts = row["timestamp"]
            mid = row["mid_price"]
            drift = row["expected_drift"]
            obi_smooth = row["obi_smooth"]
            tfi = row["tfi"]
            spread = row["spread"]

            # Detect day change and reset state
            if current_day is None:
                current_day = this_day
            elif this_day != current_day:
                # End of previous day: record closing mid price
                prev_day_close_mid = prev_mid
                # Reset cooldown state for new day
                current_day = this_day
                self.last_execution_ts = -np.inf
                self.last_limit_ts = -np.inf
                print(f"=== Starting new day: {this_day}. Previous day close mid: {prev_day_close_mid:.4f} ===")

            prev_mid = mid  # Save current mid for future day-end value

            # ---- Daily Reentry Mechanism ----
            # If flat (no position) and we've passed a small warm-up period,
            # and the new day’s mid deviates from previous day’s close by more than the daily_reentry_threshold,
            # then force a reentry.
            if this_day is not None and self.net_position == 0 and ts > 3000 and (prev_day_close_mid is not None):
                if abs(mid - prev_day_close_mid) > self.config["daily_reentry_threshold"]:
                    side_reentry = 1 if mid > prev_day_close_mid else -1
                    fill_price = row["ask_price_1"] if side_reentry > 0 else row["bid_price_1"]
                    print(f"[DAILY REENTRY] Day {this_day}: mid {mid:.2f} differs from prev close {prev_day_close_mid:.2f}")
                    self.execute_trade(this_day, ts, "daily_reentry", side_reentry, self.config["execution_unit_size"], fill_price)
                    self.last_execution_ts = ts

            # ---- Process orders/trades as usual ----
            # 1) Fill limit orders crossing the current mid price.
            self.fill_limit_orders(this_day, ts, mid)

            # 2) Check time-based exit (for open positions that linger too long with adverse signals).
            self.check_time_exit(this_day, row)

            # 3) Check trailing stop (to lock in gains if the market reverses).
            self.check_trailing_stop(this_day, row)

            # 4) Cooldown check: If the time since last execution is less than signal_decay_ms, do not enter new trades.
            if ts - self.last_execution_ts < self.config["signal_decay_ms"]:
                self.check_signal_flip(this_day, row)
                continue

            # STRATEGY A: Aggressive Market Orders.
            if (
                spread <= 2
                and abs(drift) > self.config["theta"]
                and abs(obi_smooth) > self.config["obi_threshold"]
                and np.sign(obi_smooth) == np.sign(tfi)
            ):
                side = int(np.sign(drift))
                if abs(self.net_position + side * self.config["execution_unit_size"]) <= self.config["max_position"]:
                    fill_price = row["ask_price_1"] if side > 0 else row["bid_price_1"]
                    self.execute_trade(this_day, ts, "market_order", side, self.config["execution_unit_size"], fill_price)
                    self.last_execution_ts = ts

            # STRATEGY B: Passive Limit Orders.
            elif (
                0.005 <= abs(obi_smooth) <= 0.01
                and (np.sign(obi_smooth) == np.sign(tfi) or abs(tfi) < 1e-6)
                and spread in [2, 3]
                and (ts - self.last_limit_ts >= self.config["cooldown_ms"])
            ):
                fill_prob = self.estimate_fill_probability(spread, obi_smooth)
                if fill_prob >= self.config["min_fill_probability"]:
                    side = 1 if obi_smooth > 0 else -1
                    if abs(self.net_position + side * self.config["execution_unit_size"]) <= self.config["max_position"]:
                        if spread == 3:
                            limit_price = row["ask_price_1"] if side > 0 else row["bid_price_1"]
                        else:
                            limit_price = (row["ask_price_1"] - 0.5) if side > 0 else (row["bid_price_1"] + 0.5)
                        self.place_limit_order(this_day, ts, side, self.config["execution_unit_size"], limit_price)
                        self.last_execution_ts = ts
                        self.last_limit_ts = ts

            # STRATEGY C: Impact Trigger.
            recent_trade = self.trades[
                (self.trades["day"] == this_day) & (self.trades["timestamp"] <= ts)
            ].tail(1)
            if not recent_trade.empty:
                last_vol = recent_trade.iloc[0]["quantity"]
                if (
                    last_vol >= self.config["large_trade_volume"]
                    and np.sign(obi_smooth) == np.sign(tfi)
                    and abs(drift) > self.config["theta"]
                ):
                    side = int(np.sign(tfi))
                    if abs(self.net_position + side * self.config["execution_unit_size"]) <= self.config["max_position"]:
                        fill_price = row["ask_price_1"] if side > 0 else row["bid_price_1"]
                        self.execute_trade(this_day, ts, "impact_order", side, self.config["execution_unit_size"], fill_price)
                        self.last_execution_ts = ts

            # STRATEGY D: Breakout Approach.
            self.check_breakout_signal(this_day, row)

            # 5) Check if signal flips; if so, flatten position.
            self.check_signal_flip(this_day, row)

        # Final attempt: fill any leftover limit orders at the end of the last day.
        final_day = self.prices['day'].iloc[-1]
        final_ts = self.prices['timestamp'].iloc[-1]
        final_mid = self.prices['mid_price'].iloc[-1]
        self.fill_limit_orders(final_day, final_ts + 1, final_mid)

        print(f"Strategy processed {row_count} price updates.")
        return self.execution_log

    # Helper functions

    def check_breakout_signal(self, day, row):
        ts = row["timestamp"]
        # Safety check. #todo: this can be removed, only used for testing purposes
        if "rolling_max_5" not in row or "rolling_min_5" not in row:
            return
        mid = row["mid_price"]
        rmax = row["rolling_max_5"]
        rmin = row["rolling_min_5"]
        drift = row["expected_drift"]
        obi_smooth = row["obi_smooth"]
        tfi = row["tfi"]

        if ts - self.last_execution_ts < self.config["signal_decay_ms"]:
            return

        # Up breakout condition.
        if (
            (mid > rmax + self.config["breakout_threshold_up"])
            and (drift > self.config["theta"])
            and (obi_smooth > 0)
            and (tfi > 0)
        ):
            side = +1
            if abs(self.net_position + side * self.config["execution_unit_size"]) <= self.config["max_position"]:
                fill_price = row["ask_price_1"]
                self.execute_trade(day, ts, "breakout_long", side, self.config["execution_unit_size"], fill_price)
                self.last_execution_ts = ts

        # Down breakout condition.
        elif (
            (mid < rmin - self.config["breakout_threshold_down"])
            and (drift < -self.config["theta"])
            and (obi_smooth < 0)
            and (tfi < 0)
        ):
            side = -1
            if abs(self.net_position + side * self.config["execution_unit_size"]) <= self.config["max_position"]:
                fill_price = row["bid_price_1"]
                self.execute_trade(day, ts, "breakout_short", side, self.config["execution_unit_size"], fill_price)
                self.last_execution_ts = ts

    def check_trailing_stop(self, day, row):
        """Lock in gains if price reverses after reaching a favorable peak."""
        if self.net_position == 0:
            self.best_price_in_position = None
            return

        side_held = int(np.sign(self.net_position))
        mid = row["mid_price"]

        if self.best_price_in_position is None:
            self.best_price_in_position = mid
            return

        if side_held > 0:
            # For long positions: track the highest price
            if mid > self.best_price_in_position:
                self.best_price_in_position = mid
            if (self.best_price_in_position - mid) >= self.config["trailing_stop_distance"]:
                flatten_side = -side_held
                size_to_exit = abs(self.net_position)
                fill_price = row["bid_price_1"]  # Sell at bid
                self.execute_trade(day, row["timestamp"], "trailing_stop_exit", flatten_side, size_to_exit, fill_price)
                self.best_price_in_position = None

        elif side_held < 0:
            # For short positions: track the lowest price
            if mid < self.best_price_in_position:
                self.best_price_in_position = mid
            if (mid - self.best_price_in_position) >= self.config["trailing_stop_distance"]:
                flatten_side = -side_held
                size_to_exit = abs(self.net_position)
                fill_price = row["ask_price_1"]  # Cover at ask
                self.execute_trade(day, row["timestamp"], "trailing_stop_exit", flatten_side, size_to_exit, fill_price)
                self.best_price_in_position = None

    def check_signal_flip(self, day, row):
        if self.net_position == 0:
            return
        drift = row["expected_drift"]
        side_signal = int(np.sign(drift)) if abs(drift) > self.config["theta"] else 0
        side_held = int(np.sign(self.net_position))
        if side_signal != 0 and side_signal != side_held:
            flatten_side = -side_held
            size_to_exit = abs(self.net_position)
            fill_price = row["ask_price_1"] if flatten_side > 0 else row["bid_price_1"]
            self.execute_trade(day, row["timestamp"], "signal_flip_exit", flatten_side, size_to_exit, fill_price)

    def check_time_exit(self, day, row):
        if self.net_position == 0 or self.position_entry_ts is None:
            return
        ts_now = row["timestamp"]
        if (ts_now - self.position_entry_ts) > 2000:
            drift = row["expected_drift"]
            side_signal = int(np.sign(drift)) if abs(drift) > self.config["theta"] else 0
            side_held = int(np.sign(self.net_position))
            if side_signal != side_held:
                flatten_side = -side_held
                size_to_exit = abs(self.net_position)
                fill_price = row["ask_price_1"] if flatten_side > 0 else row["bid_price_1"]
                self.execute_trade(day, ts_now, "time_stop_exit", flatten_side, size_to_exit, fill_price)

    def fill_limit_orders(self, day, now_ts, now_mid):
        for lo in self.limit_orders:
            if lo.is_filled:
                continue
            if lo.side > 0 and now_mid <= lo.limit_price:
                lo.is_filled = True
                lo.fill_ts = now_ts
                lo.fill_price = lo.limit_price
                self.execute_trade(lo.day, now_ts, "limit_fill", lo.side, lo.size, lo.fill_price)
            elif lo.side < 0 and now_mid >= lo.limit_price:
                lo.is_filled = True
                lo.fill_ts = now_ts
                lo.fill_price = lo.limit_price
                self.execute_trade(lo.day, now_ts, "limit_fill", lo.side, lo.size, lo.fill_price)

    def place_limit_order(self, day, ts, side, size, limit_price):
        lo = LimitOrder(day, ts, side, size, limit_price)
        self.limit_orders.append(lo)
        print(f"[PASSIVE] limit order placed: day={day}, side={side}, price={limit_price:.2f}, currentPos={self.net_position}")

    def execute_trade(self, day, ts, ex_type, side, size, fill_price):
        old_pos = self.net_position
        self.net_position += side * size
        if old_pos == 0 and self.net_position != 0:
            self.position_entry_ts = ts
            self.best_price_in_position = fill_price  # initialize trailing stop reference
        elif self.net_position == 0:
            self.position_entry_ts = None
            self.best_price_in_position = None

        self.execution_log.append({
            "day": day,
            "timestamp": ts,
            "type": ex_type,
            "side": side,
            "size": size,
            "price": fill_price
        })
        print(f"[TRADE] {ex_type}: day={day}, side={side}, price={fill_price:.2f}, size={size}, newPos={self.net_position}")

    def estimate_fill_probability(self, spread, obi_smooth):
        if spread == 2:
            return 0.6 if abs(obi_smooth) > 0.005 else 0.3
        elif spread == 3:
            return 0.3 if abs(obi_smooth) > 0.005 else 0.1
        return 0

    # Preprocessing methods
    def preprocess_data(self):
        """Compute mid, spread, OBI, TFI, drift, plus breakout features.
           Sort by [day, timestamp].
        """
        self.prices.sort_values(["day", "timestamp"], inplace=True)
        self.prices.reset_index(drop=True, inplace=True)

        self.compute_mid_price()
        self.compute_spread()
        self.compute_obi()
        self.compute_tfi()
        self.compute_expected_drift()
        self.compute_breakout_features()

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
        self.trades.sort_values(["day", "timestamp"], inplace=True)
        self.trades.reset_index(drop=True, inplace=True)

        self.trades["timestamp_bin"] = (self.trades["timestamp"] // 1000) * 1000
        self.trades["direction"] = np.where(
            self.trades["price"] > self.trades["price"].shift(1), 1,
            np.where(self.trades["price"] < self.trades["price"].shift(1), -1, 0)
        )
        self.trades["signed_qty"] = self.trades["direction"] * self.trades["quantity"]
        tfi_df = self.trades.groupby("timestamp_bin")["signed_qty"].sum().reset_index(name="tfi")

        self.prices["timestamp_bin"] = (self.prices["timestamp"] // 1000) * 1000
        self.prices = self.prices.merge(tfi_df, on="timestamp_bin", how="left").fillna({"tfi": 0})

    def compute_expected_drift(self):
        self.prices["future_mid"] = self.prices["mid_price"].shift(-10)
        self.prices["expected_drift"] = (self.prices["future_mid"] - self.prices["mid_price"]).fillna(0)
        max_drift = self.prices["expected_drift"].abs().max()
        self.config["theta"] = 0.5 * max_drift if max_drift > 0 else 0.1
        # print(f"Dynamic theta set to: {self.config["theta"]:.4f}")

    def compute_breakout_features(self):
        self.prices["rolling_max_5"] = self.prices["mid_price"].rolling(window=5, min_periods=1).max()
        self.prices["rolling_min_5"] = self.prices["mid_price"].rolling(window=5, min_periods=1).min()



# Main execution
if __name__ == "__main__":
    prices = pd.read_excel("C:/Users/aa04947/OneDrive - APG/Desktop/IMCR1/out/kelp_prices_combined.xlsx")
    trades = pd.read_excel("C:/Users/aa04947/OneDrive - APG/Desktop/IMCR1/out/kelp_trades_combined.xlsx")


    # cfg = StrategyConfig()
    kelp = Trader()
    logs = kelp.run()

    if not logs:
        print("No trades executed.")
    else:
        evaluator = PnLEvaluator(logs, kelp.prices)
        results_df = evaluator.compute()
        print(results_df.tail(10))  # see the last 20 lines for final net PnL
