class StrategyConfig:
    def __init__(self):
        # Spread/momentum logic
        self.theta = None  # dynamic
        self.hold_ticks = 10  # how long we hold a position
        self.max_position = 50

        # Basic microstructure thresholds
        self.obi_threshold = 0.01
        self.signal_decay_ms = 200
        self.limit_ttl_ms = 200  # limit order expires after 200 ms
        self.cooldown_ms = 100   # prevent repeated quotes

        # Sizing
        self.execution_unit_size = 1

        # Large trade impact
        self.large_trade_volume = 35

        # Passive logic
        self.min_fill_probability = 0.2