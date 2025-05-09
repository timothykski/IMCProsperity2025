class TradeRecord:
    """Stores each open position or limit order for easy tracking."""
    def __init__(self, ts, side, size, entry_price, strategy, expiry_ts=None):
        self.ts = ts
        self.side = side
        self.size = size
        self.entry_price = entry_price
        self.strategy = strategy
        self.expiry_ts = expiry_ts
        self.filled = False  # for limit orders
        self.close_ts = None
        self.close_price = None