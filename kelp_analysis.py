import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import seaborn as sns
from typing import Tuple, Optional
from scipy.stats import pearsonr

class MarketDataCombiner:
    def __init__(self, price_paths: dict, trade_paths: dict, symbol: str):
        self.price_paths = price_paths
        self.trade_paths = trade_paths
        self.symbol = symbol.upper()  # Ensure case matching
        self.prices_df = None
        self.trades_df = None

    def load_and_filter_data(self):
        prices_list = []
        trades_list = []

        for day_label, file_path in self.price_paths.items():
            df = pd.read_excel(file_path)
            df.columns = [col.lower().strip() for col in df.columns]
            df['day'] = int(day_label.split()[-1])  # Extract 0, 1, 2 from "Day X"
            df = df[df['product'].str.upper() == self.symbol]
            prices_list.append(df)

        for day_label, file_path in self.trade_paths.items():
            df = pd.read_excel(file_path)
            df.columns = [col.lower().strip() for col in df.columns]
            df['day'] = int(day_label.split()[-1])
            df = df[df['symbol'].str.upper() == self.symbol]
            trades_list.append(df)

        self.prices_df = pd.concat(prices_list, ignore_index=True).sort_values(by=['day', 'timestamp'])
        self.trades_df = pd.concat(trades_list, ignore_index=True).sort_values(by=['day', 'timestamp'])

        return self.prices_df, self.trades_df

    def export_to_excel(self, output_dir: str):
        # optional, not necessary
        os.makedirs(output_dir, exist_ok=True)
        prices_path = os.path.join(output_dir, f"kelp_prices_combined.xlsx")
        trades_path = os.path.join(output_dir, f"kelp_trades_combined.xlsx")
        self.prices_df.to_excel(prices_path, index=False)
        self.trades_df.to_excel(trades_path, index=False)
        print(f"Saved combined prices to: {prices_path}")
        print(f"Saved combined trades to: {trades_path}")

class OrderBookAnalytics:
    def __init__(self, prices_df: pd.DataFrame, trades_df: pd.DataFrame):
        self.prices = prices_df.copy().reset_index(drop=True)
        self.trades = trades_df.copy().reset_index(drop=True)
        self.analytics_df = self.prices.copy()

    def compute_mid_price(self):
        self.analytics_df["mid_price"] = (
            self.analytics_df["bid_price_1"] + self.analytics_df["ask_price_1"]
        ) / 2

    def compute_rolling_volatility(self, window: int = 50):
        self.analytics_df["rolling_volatility"] = (
            self.analytics_df["mid_price"].rolling(window=window).std()
        )

    def compute_spread(self):
        self.analytics_df["spread"] = (
            self.analytics_df["ask_price_1"] - self.analytics_df["bid_price_1"]
        )

    def compute_order_book_imbalance(self):
        bid_vols = sum(self.analytics_df[f"bid_volume_{i}"].fillna(0) for i in range(1, 4))
        ask_vols = sum(self.analytics_df[f"ask_volume_{i}"].fillna(0) for i in range(1, 4))
        self.analytics_df["obi"] = (bid_vols - ask_vols) / (bid_vols + ask_vols + 1e-6)

    def run_all(self):
        self.compute_mid_price()
        self.compute_rolling_volatility()
        self.compute_spread()
        self.compute_order_book_imbalance()
        return self.analytics_df

class OrderBookPredictiveAnalysis:
    def __init__(self, analytics_df: pd.DataFrame, trades_df: pd.DataFrame):
        self.df = analytics_df.copy()
        self.trades = trades_df.copy()

        # Check essential columns exist
        for col in ["spread", "obi", "mid_price"]:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        self.days = sorted(self.df["day"].unique())

    def autocorrelation_analysis(self, lags: int = 50):
        results = {}
        for day in self.days:
            df_day = self.df[self.df["day"] == day]
            ac = {
                col: [df_day[col].dropna().autocorr(lag=i) for i in range(1, lags + 1)]
                for col in ["spread", "obi"]
            }
            results[day] = ac
        return results

    def conditional_mid_price_drift(self, spread_bins=[1, 2, 3, 4], obi_bins=np.linspace(-0.02, 0.02, 9), horizon=50):
        results = {}
        for day in self.days:
            df = self.df[self.df["day"] == day].copy()
            df["future_mid"] = df["mid_price"].shift(-horizon)
            df["mid_drift"] = df["future_mid"] - df["mid_price"]

            res = []
            for s in spread_bins:
                for i in range(len(obi_bins) - 1):
                    obi_min, obi_max = obi_bins[i], obi_bins[i + 1]
                    subset = df[(df["spread"] == s) & (df["obi"] >= obi_min) & (df["obi"] < obi_max)]
                    if not subset.empty:
                        drift_mean = subset["mid_drift"].mean()
                        res.append({
                            "spread": s,
                            "obi_bin_center": (obi_min + obi_max) / 2,
                            "avg_drift": drift_mean,
                            "count": len(subset)
                        })
            results[day] = pd.DataFrame(res)
        return results

    def estimate_trade_direction(self):
        results = {}
        for day in self.days:
            trades = self.trades[self.trades["day"] == day].copy()
            prices = self.df[self.df["day"] == day].copy()

            trades["timestamp_bin"] = (trades["timestamp"] // 1000) * 1000
            prices["timestamp_bin"] = (prices["timestamp"] // 1000) * 1000

            merged = pd.merge_asof(
                trades.sort_values("timestamp"),
                prices[["timestamp_bin", "bid_price_1", "ask_price_1"]].sort_values("timestamp_bin"),
                on="timestamp_bin",
                direction="nearest"
            )

            merged["direction"] = np.where(
                np.abs(merged["price"] - merged["bid_price_1"]) < np.abs(merged["price"] - merged["ask_price_1"]),
                -1, 1
            )
            merged["signed_qty"] = merged["direction"] * merged["quantity"]
            flow = merged.groupby("timestamp_bin")["signed_qty"].sum().reset_index(name="flow_imbalance")
            results[day] = flow
        return results

    def compute_execution_impact(self, volume_threshold=20, horizon=30):
        results = {}
        for day in self.days:
            df = self.df[self.df["day"] == day].copy()
            trades = self.trades[self.trades["day"] == day].copy()

            df["mid_price_fwd"] = df["mid_price"].shift(-horizon)
            df["mid_return"] = df["mid_price_fwd"] - df["mid_price"]

            volume = trades.groupby((trades["timestamp"] // 1000) * 1000)["quantity"].sum().reset_index()
            volume.columns = ["timestamp_bin", "volume"]

            df["timestamp_bin"] = (df["timestamp"] // 1000) * 1000
            merged = pd.merge(df, volume, how="left", on="timestamp_bin")
            merged["volume"] = merged["volume"].fillna(0)

            heavy = merged[merged["volume"] >= volume_threshold]
            results[day] = heavy[["timestamp", "volume", "mid_return"]]
        return results

    def compute_execution_impact_profile(self, volume_bins=range(20, 50, 5), horizon=30):
        results = {}
        for day in self.days:
            df = self.df[self.df["day"] == day].copy()
            trades = self.trades[self.trades["day"] == day].copy()

            df["mid_price_fwd"] = df["mid_price"].shift(-horizon)
            df["mid_return"] = df["mid_price_fwd"] - df["mid_price"]
            df["timestamp_bin"] = (df["timestamp"] // 1000) * 1000

            volume = trades.groupby((trades["timestamp"] // 1000) * 1000)["quantity"].sum().reset_index()
            volume.columns = ["timestamp_bin", "volume"]

            merged = pd.merge(df, volume, how="left", on="timestamp_bin")
            merged["volume"] = merged["volume"].fillna(0)

            stats = []
            for i in range(len(volume_bins) - 1):
                vmin, vmax = volume_bins[i], volume_bins[i + 1]
                bin_df = merged[(merged["volume"] >= vmin) & (merged["volume"] < vmax)]
                if not bin_df.empty:
                    mean_ret = bin_df["mid_return"].mean()
                    std_ret = bin_df["mid_return"].std()
                    stats.append({
                        "volume_bin": f"{vmin}-{vmax}",
                        "mean_return": mean_ret,
                        "std_return": std_ret
                    })
            results[day] = pd.DataFrame(stats)
        return results

    def plot_autocorrelation(self, ac_results):
        fig, axs = plt.subplots(len(self.days), 1, figsize=(10, 4 * len(self.days)), sharex=True)
        for i, day in enumerate(self.days):
            axs[i].plot(ac_results[day]["spread"], label="Spread")
            axs[i].plot(ac_results[day]["obi"], label="OBI")
            axs[i].set_title(f"Autocorrelation — Day {day}")
            axs[i].set_xlabel("Lag")
            axs[i].set_ylabel("Autocorrelation")
            axs[i].grid(True)
            axs[i].legend()
        plt.tight_layout()
        plt.show()

    def plot_drift_heatmap(self, drift_results):
        fig, axs = plt.subplots(len(self.days), 1, figsize=(10, 5 * len(self.days)))
        for i, day in enumerate(self.days):
            pivot = drift_results[day].pivot(index="obi_bin_center", columns="spread", values="avg_drift")
            sns.heatmap(pivot, ax=axs[i], cmap="coolwarm", center=0, annot=True)
            axs[i].set_title(f"Conditional Mid-Price Drift — Day {day}")
            axs[i].set_xlabel("Spread")
            axs[i].set_ylabel("OBI Bin Center")
        plt.tight_layout()
        plt.show()

    def plot_trade_flow(self, flow_results):
        fig, axs = plt.subplots(len(self.days), 1, figsize=(12, 3.5 * len(self.days)), sharex=False)
        for i, day in enumerate(self.days):
            df = flow_results[day]
            axs[i].plot(df["timestamp_bin"], df["flow_imbalance"], color="purple")
            axs[i].set_title(f"Trade Flow Imbalance — Day {day}")
            axs[i].set_xlabel("Timestamp")
            axs[i].set_ylabel("Signed Volume")
            axs[i].grid(True)
        plt.tight_layout()
        plt.show()

    def plot_execution_impact(self, impact_results):
        fig, axs = plt.subplots(len(self.days), 1, figsize=(10, 4 * len(self.days)))
        for i, day in enumerate(self.days):
            df = impact_results[day]
            sns.scatterplot(data=df, x="volume", y="mid_return", alpha=0.6, ax=axs[i])
            axs[i].set_title(f"Execution Impact — Day {day}")
            axs[i].set_xlabel("Volume")
            axs[i].set_ylabel("Mid Price Change")
            axs[i].grid(True)
        plt.tight_layout()
        plt.show()

    def plot_execution_impact_profile(self, profile_results):
        fig, axs = plt.subplots(len(self.days), 1, figsize=(10, 4 * len(self.days)))
        for i, day in enumerate(self.days):
            df = profile_results[day]
            if not df.empty:
                axs[i].bar(df["volume_bin"], df["mean_return"], yerr=df["std_return"],
                           capsize=4, alpha=0.7, color="steelblue")
                axs[i].set_title(f"Execution Impact Profile — Day {day}")
                axs[i].set_ylabel("Avg Mid-Price Change")
                axs[i].grid(True)
            else:
                axs[i].set_title(f"Execution Impact Profile — Day {day} (No Data)")
                axs[i].grid(True)

        axs[-1].set_xlabel("Trade Volume Bins")
        plt.tight_layout()
        plt.show()

    def run_all_and_plot(self):
        ac = self.autocorrelation_analysis()
        drift = self.conditional_mid_price_drift()
        flow = self.estimate_trade_direction()
        impact = self.compute_execution_impact()
        impact_profile = self.compute_execution_impact_profile()

        self.plot_autocorrelation(ac)
        self.plot_drift_heatmap(drift)
        self.plot_trade_flow(flow)
        self.plot_execution_impact(impact)
        self.plot_execution_impact_profile(impact_profile)

        return {
            "autocorrelation": ac,
            "drift_by_spread_obi": drift,
            "trade_flow_imbalance": flow,
            "execution_impact": impact,
            "execution_impact_profile": impact_profile
        }


class OrderBookPlots:
    def __init__(self, analytics_df: pd.DataFrame, trades_df: pd.DataFrame):
        self.df = analytics_df
        self.trades = trades_df

    def _plot_by_day(self, y_col: str, title: str, ylabel: str, smooth: bool = False):
        days = sorted(self.df['day'].unique())
        num_days = len(days)

        fig, axs = plt.subplots(num_days, 1, figsize=(12, 4 * num_days), sharex=False)
        axs = np.atleast_1d(axs)

        for idx, day in enumerate(days):
            df_day = self.df[self.df["day"] == day].copy()
            x = df_day["timestamp"]
            y = df_day[y_col]

            if smooth and not y_col.startswith("rolling"):
                y = y.rolling(100).mean()

            axs[idx].plot(x, y, label=y_col)
            axs[idx].set_title(f"{title} — Day {day}")
            axs[idx].set_xlabel("Timestamp")
            axs[idx].set_ylabel(ylabel)
            axs[idx].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_mid_price(self):
        self._plot_by_day("mid_price", "Mid-Price vs. Timestamp", "Mid Price")

    def plot_mid_price_with_spread(self, window: int = 100):
        days = sorted(self.df["day"].unique())
        num_days = len(days)

        fig, axs = plt.subplots(num_days, 1, figsize=(14, 4 * num_days), sharex=False)
        axs = np.atleast_1d(axs)

        for idx, day in enumerate(days):
            df_day = self.df[self.df["day"] == day].copy()

            # Compute rolling average of spread to reduce visual noise
            df_day["rolling_spread"] = df_day["spread"].rolling(window=window, min_periods=1).mean()

            axs[idx].plot(df_day["timestamp"], df_day["mid_price"], label="Mid Price", color="black", linewidth=1)

            # Create color mask by spread level
            colors = {
                1: "green",  # tightest
                2: "yellow",
                3: "orange",
                4: "red"  # widest
            }

            for spread_value, color in colors.items():
                mask = df_day["spread"] == spread_value
                axs[idx].fill_between(df_day["timestamp"],
                                      df_day["mid_price"].min(),
                                      df_day["mid_price"].max(),
                                      where=mask,
                                      color=color,
                                      alpha=0.03,
                                      label=f"Spread = {spread_value}")

            axs[idx].set_title(f"Mid-Price with Spread Heatmap — Day {day}")
            axs[idx].set_xlabel("Timestamp")
            axs[idx].set_ylabel("Mid Price")
            axs[idx].legend()
            axs[idx].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_rolling_volatility(self):
        self._plot_by_day("rolling_volatility", "Rolling Volatility of Mid-Price", "Volatility")

    def plot_spread(self, bin_size: int = 10000):
        days = sorted(self.df["day"].unique())
        num_days = len(days)

        fig, axs = plt.subplots(num_days, 1, figsize=(12, 3.5 * num_days), sharex=False)
        axs = np.atleast_1d(axs)

        for idx, day in enumerate(days):
            df_day = self.df[self.df["day"] == day].copy()
            df_day["bin"] = (df_day["timestamp"] // bin_size) * bin_size
            binned = df_day.groupby("bin")["spread"].mean().reset_index()

            axs[idx].plot(binned["bin"], binned["spread"])
            axs[idx].set_title(f"Avg Bid-Ask Spread Over Time — Day {day}")
            axs[idx].set_xlabel("Timestamp (binned)")
            axs[idx].set_ylabel("Spread")
            axs[idx].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_spread_distribution_by_day(self):
        plt.figure(figsize=(10, 6))

        days = sorted(self.df["day"].unique())
        colors = sns.color_palette("tab10", len(days))

        for i, day in enumerate(days):
            subset = self.df[self.df["day"] == day]
            sns.histplot(
                subset["spread"],
                bins=[0.5, 1.5, 2.5, 3.5, 4.5],
                kde=True,
                stat="density",
                label=f"Day {day}",
                color=colors[i],
                edgecolor="black",
                alpha=0.4,
                linewidth=1
            )

        plt.title("Bid-Ask Spread Distribution per Day (with KDE)")
        plt.xlabel("Spread (in ticks)")
        plt.ylabel("Density")
        plt.xticks([1, 2, 3, 4])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_order_book_imbalance(self):
        self._plot_by_day("obi", "Order Book Imbalance Over Time", "Imbalance (OBI)", smooth=True)

    def plot_trade_volume(self, bin_size: int = 100):
        self.trades["bin"] = (self.trades["timestamp"] // bin_size) * bin_size
        days = sorted(self.trades["day"].unique())
        num_days = len(days)

        fig, axs = plt.subplots(num_days, 1, figsize=(12, 3.5 * num_days), sharex=False)
        axs = np.atleast_1d(axs)

        for idx, day in enumerate(days):
            day_trades = self.trades[self.trades["day"] == day]
            volume = day_trades.groupby("bin")["quantity"].sum().reset_index()
            axs[idx].plot(volume["bin"], volume["quantity"])
            axs[idx].set_title(f"Trade Volume Over Time — Day {day}")
            axs[idx].set_xlabel("Timestamp (binned)")
            axs[idx].set_ylabel("Volume")
            axs[idx].grid(True)

        plt.tight_layout()
        plt.show()

    def run_all_plots(self):
        self.plot_mid_price()
        self.plot_mid_price_with_spread()
        self.plot_rolling_volatility()
        self.plot_order_book_imbalance()
        self.plot_spread()
        self.plot_spread_distribution_by_day()
        self.plot_trade_volume()



if __name__ == "__main__":
    # Define file paths
    # prices_paths = {
    #     "Day 0": r"C:/Users/aa04947/OneDrive - APG/Desktop/IMCR1/out/prices_day_0_cleaned.xlsx",
    #     "Day 1": r"C:/Users/aa04947/OneDrive - APG/Desktop/IMCR1/out/prices_day_-1_cleaned.xlsx",
    #     "Day 2": r"C:/Users/aa04947/OneDrive - APG/Desktop/IMCR1/out/prices_day_-2_cleaned.xlsx",
    # }
    #
    # trades_paths = {
    #     "Day 0": r"C:/Users/aa04947/OneDrive - APG/Desktop/IMCR1/out/trades_day_0_cleaned.xlsx",
    #     "Day 1": r"C:/Users/aa04947/OneDrive - APG/Desktop/IMCR1/out/trades_day_-1_cleaned.xlsx",
    #     "Day 2": r"C:/Users/aa04947/OneDrive - APG/Desktop/IMCR1/out/trades_day_-2_cleaned.xlsx",
    # }

    # Define the symbol to focus on
    target_symbol = "KELP"

    # Run combiner
    # combiner = MarketDataCombiner(prices_paths, trades_paths, target_symbol)
    # kelp_prices_df, kelp_trades_df = combiner.load_and_filter_data()
    kelp_prices_df = pd.read_excel(r"C:/Users/aa04947/OneDrive - APG/Desktop/IMCR1/out/kelp_prices_combined.xlsx")
    kelp_trades_df = pd.read_excel(r"C:/Users/aa04947/OneDrive - APG/Desktop/IMCR1/out/kelp_trades_combined.xlsx")

    # Step 1: Compute analytics
    oba = OrderBookAnalytics(kelp_prices_df, kelp_trades_df)
    analytics_df = oba.run_all()

    # Step 2: Plot everything
    plotter = OrderBookPlots(analytics_df, kelp_trades_df)
    plotter.run_all_plots()

    # Output previews
    # print("KELP PRICES DATA (Combined):")
    print(kelp_prices_df.head(), "\n")

    # print("KELP TRADES DATA (Combined):")
    print(kelp_trades_df.head())

    predictive = OrderBookPredictiveAnalysis(analytics_df, kelp_trades_df)
    results = predictive.run_all_and_plot()


    # combiner.export_to_excel(r"C:/Users/aa04947/OneDrive - APG/Desktop/IMCR1/out")
