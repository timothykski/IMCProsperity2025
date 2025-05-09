# Data Structure
class LimitOrder:
    """Represent a resting limit order in the order book, uncancelable."""

    def __init__(self, day, ts, side, size, limit_price):
        self.day = day
        self.ts = ts
        self.side = side  # +1 = buy, -1 = sell
        self.size = size
        self.limit_price = limit_price
        self.is_filled = False
        self.fill_ts = None
        self.fill_price = None


class ExecutionRecord:
    """Track each fill event for PnL analysis."""

    def __init__(self, day, timestamp, ex_type, side, size, price):
        self.day = day
        self.timestamp = timestamp
        self.type = ex_type  # "market_order", "limit_fill", "impact_order", etc.
        self.side = side
        self.size = size
        self.price = price