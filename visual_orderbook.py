import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class OrderBookAnimator:
    def __init__(self, file_path: str, product: str, day_label: str):
        self.file_path = file_path
        self.product = product
        self.day_label = day_label
        self.df = self._load_and_filter_data()

    def _load_and_filter_data(self) -> pd.DataFrame:
        df = pd.read_excel(self.file_path)
        df.columns = [col.lower().strip() for col in df.columns]
        df['timestamp'] = df['timestamp'].astype(int)
        df = df[df['product'] == self.product].sort_values('timestamp').reset_index(drop=True)
        return df

    def animate(self):
        fig, ax = plt.subplots(figsize=(10, 6))

        def update(frame):
            ax.clear()
            snapshot = self.df.iloc[frame]

            bid_prices, bid_volumes = [], []
            ask_prices, ask_volumes = [], []

            best_bid = snapshot.get('bid_price_1')
            best_ask = snapshot.get('ask_price_1')
            mid_price = (best_bid + best_ask) / 2 if pd.notna(best_bid) and pd.notna(best_ask) else None

            for level in [1, 2, 3]:
                bp, bv = snapshot.get(f'bid_price_{level}'), snapshot.get(f'bid_volume_{level}')
                ap, av = snapshot.get(f'ask_price_{level}'), snapshot.get(f'ask_volume_{level}')
                if pd.notna(bp) and pd.notna(bv):
                    bid_prices.append(bp)
                    bid_volumes.append(bv)
                if pd.notna(ap) and pd.notna(av):
                    ask_prices.append(ap)
                    ask_volumes.append(av)

            ax.bar(bid_prices, bid_volumes, color='blue', label='Bid Side', width=0.4)
            ax.bar(ask_prices, ask_volumes, color='red', label='Ask Side', width=0.4)

            if mid_price:
                ax.axvline(mid_price, color='black', linestyle='--', label=f"Mid Price ({mid_price:.2f})")

            ax.set_title(f"Order Book Snapshot - {self.product} - {self.day_label} - t={snapshot['timestamp']}")
            ax.set_xlabel("Prices")
            ax.set_ylabel("Depth")
            ax.legend()

            all_prices = bid_prices + ask_prices + ([mid_price] if mid_price else [])
            all_volumes = bid_volumes + ask_volumes
            if all_prices:
                ax.set_xlim(min(all_prices) - 2, max(all_prices) + 2)
            if all_volumes:
                ax.set_ylim(0, max(all_volumes) + 5)

        ani = animation.FuncAnimation(fig, update, frames=len(self.df), interval=300)
        plt.show()

class MidPricePlotter:
    def __init__(self, file_path: str, day_label: str):
        self.file_path = file_path
        self.day_label = day_label
        self.df = self._load_and_prepare()

    def _load_and_prepare(self):
        df = pd.read_excel(self.file_path)
        df.columns = [col.lower().strip() for col in df.columns]
        df['timestamp'] = df['timestamp'].astype(int)
        return df

    def plot_all_symbols(self):
        for symbol in self.df['product'].unique():
            self.plot_single_symbol(symbol)

    def plot_single_symbol(self, symbol: str):
        symbol_df = self.df[self.df['product'] == symbol].copy()
        symbol_df['mid_price'] = (symbol_df['bid_price_1'] + symbol_df['ask_price_1']) / 2

        plt.figure(figsize=(10, 5))
        plt.plot(symbol_df['timestamp'], symbol_df['mid_price'], label='Mid Price', color='black')
        plt.title(f"Mid Price vs. Timestamp - {symbol} - {self.day_label}")
        plt.xlabel("Timestamp")
        plt.ylabel("Mid Price")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# === Main Section ===
if __name__ == "__main__":
    paths = {
        "Day 0": r"C:/Users/aa04947/OneDrive - APG/Desktop/IMCR1/out/prices_day_0_cleaned.xlsx",
        "Day 1": r"C:/Users/aa04947/OneDrive - APG/Desktop/IMCR1/out/prices_day_1_cleaned.xlsx",
        "Day 2": r"C:/Users/aa04947/OneDrive - APG/Desktop/IMCR1/out/prices_day_2_cleaned.xlsx",
    }

    product = "KELP"
    animator = OrderBookAnimator(paths["Day 0"], product, "Day 0")
    animator.animate()

    for day_label, path in paths.items():
        print(f"Plotting for {day_label}")
        plotter = MidPricePlotter(path, day_label)
        plotter.plot_all_symbols()
