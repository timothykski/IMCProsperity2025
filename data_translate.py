import pandas as pd


def load_prices_data(file_path: str) -> pd.DataFrame:
    """
    Loads and cleans prices data from CSV.
    """
    prices_df = pd.read_csv(file_path, delimiter=";")
    prices_df.columns = [col.strip().lower() for col in prices_df.columns]
    prices_df['timestamp'] = prices_df['timestamp'].astype(int)

    # Optional: sort by time/product
    prices_df = prices_df.sort_values(by=['product', 'timestamp']).reset_index(drop=True)

    return prices_df


def load_trades_data(file_path: str) -> pd.DataFrame:
    """
    Loads and cleans trades data from CSV.
    """
    trades_df = pd.read_csv(file_path, delimiter=";")
    trades_df.columns = [col.strip().lower() for col in trades_df.columns]
    trades_df['timestamp'] = trades_df['timestamp'].astype(int)

    # Optional: sort by time/product
    trades_df = trades_df.sort_values(by=['symbol', 'timestamp']).reset_index(drop=True)

    return trades_df


def describe_prices_schema(df: pd.DataFrame):
    print("Prices Data Columns:")
    print("- timestamp: when the order book snapshot was taken")
    print("- product: name of the commodity (e.g. KELP, RAINFOREST_RESIN)")
    print("- bid_price_X / ask_price_X: top 3 bid/ask prices")
    print("- bid_volume_X / ask_volume_X: volumes available at each price level")
    print("- mid_price: midpoint of best bid and ask")
    print("- profit_and_loss: your team's cumulative PnL (optional or placeholder)\n")


def describe_trades_schema(df: pd.DataFrame):
    print("Trades Data Columns:")
    print("- timestamp: when the trade occurred")
    print("- symbol: name of the traded commodity")
    print("- price: actual transaction price")
    print("- quantity: trade size")
    print("- buyer / seller: blank for now, may be used in future rounds\n")


# main execution
if __name__ == "__main__":
    prices_path = r"C:/Users/aa04947/OneDrive - APG/Desktop/IMCR1/prices_round_1_day_-2.csv"
    trades_path = r"C:/Users/aa04947/OneDrive - APG/Desktop/IMCR1/trades_round_1_day_-2.csv"

    prices_df = load_prices_data(prices_path)
    trades_df = load_trades_data(trades_path)

    # Export cleaned DataFrames to Excel under if __name__ == "__main__"
    prices_df.to_excel(r"C:/Users/aa04947/OneDrive - APG/Desktop/IMCR1/out/prices_day_2_cleaned.xlsx", index=False)
    trades_df.to_excel(r"C:/Users/aa04947/OneDrive - APG/Desktop/IMCR1/out/trades_day_2_cleaned.xlsx", index=False)

    describe_prices_schema(prices_df)
    describe_trades_schema(trades_df)

    # Preview
    print("Preview of Cleaned Prices:")
    print(prices_df, "\n")

    print("Preview of Cleaned Trades:")
    print(trades_df)
