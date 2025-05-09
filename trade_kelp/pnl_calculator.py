import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from order import LimitOrder, ExecutionRecord

# PnL Evaluation, calculation
class PnLEvaluator:
    """
    1) Rebuild a trade-by-trade ledger to know exactly how position changes over time.
    2) For each price row, we track:
        - Current net position
        - Realized PnL from trades that reduce position
        - Unrealized PnL from net_position * (current_mid_price - average_cost)
    3) Total PnL = realized + unrealized PnL of the last price.
    """

    def __init__(self, logs: list, prices_df: pd.DataFrame):
        self.execs = pd.DataFrame(logs)
        self.prices = prices_df.copy().reset_index(drop=True)

        # Arrays to accumulate PnL over time
        self.times = []          # We'll store row indices (1 to N)
        self.realized_pnl = []
        self.unrealized_pnl = []
        self.cum_pnl = []

    def compute(self):
        if self.execs.empty:
            print("No trades executed. PnL = 0.")
            return pd.DataFrame()

        # Sort by day first and then by timestamp to ensure proper chronological order
        self.execs.sort_values(["day", "timestamp"], inplace=True)
        self.prices.sort_values(["day", "timestamp"], inplace=True)

        # For incremental PnL tracking
        current_position = 0
        avg_cost = 0.0  # Weighted average cost (for the position we hold)
        total_realized = 0.0

        price_idx = 0
        price_len = len(self.prices)

        trade_idx = 0
        n_trades = len(self.execs)

        # Convert DataFrame of trades into list (faster for iteration)
        trades_list = self.execs.to_dict('records')

        results_rows = []

        while price_idx < price_len:
            # Instead of raw timestamp, we use row index (1-based) for plotting.
            row_index = price_idx + 1

            rowP = self.prices.iloc[price_idx]
            now_ts = rowP['timestamp']  # still used for time-based comparisons
            now_mid = rowP['mid_price']

            # Process any trades at or before now_ts
            while trade_idx < n_trades and trades_list[trade_idx]['timestamp'] <= now_ts:
                tr = trades_list[trade_idx]
                side = tr['side']
                size = tr['size']
                fill_px = tr['price']

                old_position = current_position
                old_value = current_position * avg_cost
                new_position = current_position + side * size

                # Update position and calculate realized PnL if reducing or flipping position
                if np.sign(new_position) == np.sign(old_position):
                    # Same side or coming out of zero: update average cost
                    if old_position == 0:
                        avg_cost = fill_px
                    else:
                        new_value = old_value + side * size * fill_px
                        if new_position != 0:
                            avg_cost = new_value / new_position
                        else:
                            avg_cost = 0.0
                    current_position = new_position
                else:
                    # Reducing or flipping position: realize PnL
                    if abs(new_position) < abs(old_position):
                        # Partial flatten: realize PnL on traded portion.
                        realized_part = side * size * (fill_px - avg_cost)
                        total_realized += realized_part
                        current_position = new_position
                    else:
                        # Full flatten or flip: realize PnL on full old position.
                        realized_part = -old_position * (fill_px - avg_cost)
                        total_realized += realized_part
                        current_position = new_position
                        leftover_size = abs(new_position)
                        if leftover_size > 0:
                            avg_cost = fill_px
                        else:
                            avg_cost = 0.0

                trade_idx += 1

            # Mark-to-market calculation: Unrealized PnL = current_position * (now_mid - avg_cost)
            ur = current_position * (now_mid - avg_cost)
            cum = total_realized + ur

            self.times.append(row_index)  # Using row index as x-axis
            self.realized_pnl.append(total_realized)
            self.unrealized_pnl.append(ur)
            self.cum_pnl.append(cum)

            results_rows.append({
                "row_index": row_index,
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
        plt.plot(self.times, self.cum_pnl, label="Total Cumulative PnL", linewidth=1.5, color="blue")
        plt.plot(self.times, self.realized_pnl, label="Realized PnL", linestyle="--", linewidth=1.5, color="orange")
        plt.axhline(0, color="gray", linestyle="--")
        plt.xlabel("Row Index (1...N)")
        plt.ylabel("PnL")
        plt.title("Mark-to-Market & Realized PnL (3 Days Combined)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()