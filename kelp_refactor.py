from datamodel import Order, OrderDepth, Trade, TradingState
import math
import numpy as np
from collections import deque

# Declare product constants for IMC environment
KELP = "KELP"
PRODUCTS = [KELP]


class Trader:
    def __init__(self):
        """
        This mirrors your original config, with maximum position, thresholds, etc.
        We store rolling data for mid prices, OBI, TFI, etc. so we can replicate
        your logic that used to rely on a pandas DataFrame.
        """
        self.config = {
            "theta": None,
            "drift_size_map": [    # (low_drift, high_drift, (low_size, high_size)) if you want to expand usage
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

        # Positioning and internal state
        self.net_position = 0
        self.position_entry_ts = None  # used for time-based exit
        self.last_execution_ts = -999999999
        self.last_limit_ts = -999999999
        self.best_price_in_position = None

        # We track the “current day” for daily reentry logic
        self.current_day = None
        # Store the last day’s closing mid price
        self.prev_day_close_mid = None

        # Rolling storage of mid prices for drift, etc.
        self.mid_prices = deque(maxlen=2000)   # store as many as you like
        # Rolling storage for OBI computations
        self.obi_values = deque(maxlen=20)
        # Rolling storage for TFI computations
        self.tfi_values = deque(maxlen=50)

        # Will set self.config["theta"] dynamically from “drift” logic, if desired
        self.config["theta"] = 0.1  # fallback default

    def run(self, state: TradingState):
        """
        The main method, called once per tick. We replicate the structure
        of your old loop over each row, but now adapt it to real-time usage.
        The logic is basically:
         1) Identify if the day has changed => daily reentry reset
         2) Compute mid, drift, OBI, TFI from the current order book + trades
         3) Check trailing stops, time-based exit
         4) Check cooldown
         5) Run your strategy: A/B/C triggers, breakout approach, check signal flip
         6) Return a dictionary of Orders to place for KELP
        """

        orders_to_place = []

        # 1) Identify the “day” based on timestamp, e.g. day = timestamp // 100000
        day = state.timestamp // 100000
        if self.current_day is None:
            self.current_day = day
        elif day != self.current_day:
            # We treat it like a new day
            print(f"=== Starting new day: {day}. Prev day close mid: {self.prev_day_close_mid}")
            self.current_day = day
            self.last_execution_ts = -999999999
            self.last_limit_ts = -999999999

        # 2) Extract data from the order book for KELP
        if KELP not in state.order_depths:
            # No market data => no trading
            return {KELP: []}, 1, None

        order_depth: OrderDepth = state.order_depths[KELP]

        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            # If we can’t compute a mid price, skip
            return {KELP: []}, 1, None

        mid = (best_bid + best_ask)/2

        # Gather real-time OBI and TFI from the data
        obi_raw = self.compute_obi(order_depth)
        tfi_raw = self.compute_tfi(state.market_trades.get(KELP, []), state.timestamp)

        # Store them in rolling arrays so we can do smoothing, etc.
        self.mid_prices.append(mid)
        self.obi_values.append(obi_raw)
        self.tfi_values.append(tfi_raw)

        # Possibly we do a rolling average for obi_smooth:
        obi_smooth = sum(self.obi_values)/len(self.obi_values)

        # For drift, we can replicate your “expected_drift” by looking 10 ticks back, or do something simpler
        drift = self.compute_drift()

        # Update dynamic theta if you like
        self.update_theta(drift)

        # We also track spread
        spread = best_ask - best_bid

        # Update net position from the state
        self.net_position = state.position.get(KELP, 0)

        # 2B) Daily Reentry Mechanism if net_position == 0, after warm-up time
        # We have to guess if we want warm-up to be e.g. timestamp>3000
        # We also compare mid vs self.prev_day_close_mid if that is not None
        if day != self.current_day:
            # that means we switched days, so store the old close mid
            self.prev_day_close_mid = mid

        if (
            self.net_position == 0
            and state.timestamp > 3000
            and self.prev_day_close_mid is not None
        ):
            if abs(mid - self.prev_day_close_mid) > self.config["daily_reentry_threshold"]:
                side_reentry = 1 if mid > self.prev_day_close_mid else -1
                fill_price = best_ask if side_reentry > 0 else best_bid
                print(f"[DAILY REENTRY] day={day}, mid={mid:.2f} differs from prev close={self.prev_day_close_mid:.2f}")
                self.execute_trade(side_reentry, self.config["execution_unit_size"], fill_price, orders_to_place)
                self.last_execution_ts = state.timestamp

        # 3) Check time-based exit, trailing stops
        self.check_time_exit(state.timestamp, drift, best_bid, best_ask)
        self.check_trailing_stop(mid, best_bid, best_ask, state.timestamp)

        # 4) Check cooldown
        if (state.timestamp - self.last_execution_ts) < self.config["signal_decay_ms"]:
            # If we are in cooldown, we at least check signal flip
            self.check_signal_flip(drift, best_bid, best_ask, state.timestamp, orders_to_place)
            return {KELP: orders_to_place}, 1, None

        # 5) STRATEGY A: Aggressive Market Orders
        if (
            spread <= 2
            and abs(drift) > (self.config["theta"] or 0.01)
            and abs(obi_smooth) > self.config["obi_threshold"]
            and np.sign(obi_smooth) == np.sign(tfi_raw)
        ):
            side = int(np.sign(drift))
            candidate_pos = self.net_position + side * self.config["execution_unit_size"]
            if abs(candidate_pos) <= self.config["max_position"]:
                # “Market” means crossing the spread: buy at best_ask or sell at best_bid
                fill_price = best_ask if side>0 else best_bid
                self.execute_trade(side, self.config["execution_unit_size"], fill_price, orders_to_place)
                self.last_execution_ts = state.timestamp

        # STRATEGY B: Passive Limit Orders
        elif (
            0.005 <= abs(obi_smooth) <= 0.01
            and (np.sign(obi_smooth) == np.sign(tfi_raw) or abs(tfi_raw) < 1e-6)
            and spread in [2, 3]
            and (state.timestamp - self.last_limit_ts >= self.config["cooldown_ms"])
        ):
            fill_prob = self.estimate_fill_probability(spread, obi_smooth)
            if fill_prob >= self.config["min_fill_probability"]:
                side = 1 if obi_smooth > 0 else -1
                candidate_pos = self.net_position + side*self.config["execution_unit_size"]
                if abs(candidate_pos) <= self.config["max_position"]:
                    if spread == 3:
                        # place limit at best_ask if side>0 or best_bid if side<0
                        limit_price = best_ask if side>0 else best_bid
                    else:
                        # spread==2 => place slightly inside?
                        limit_price = (best_ask - 0.5) if side>0 else (best_bid + 0.5)
                    # We can place a limit order by simply quoting it:
                    # The environment might or might not fill us
                    orders_to_place.append(Order(KELP, int(limit_price), side*self.config["execution_unit_size"]))
                    print(f"[PASSIVE] limit order placed side={side}, price={limit_price}, currentPos={self.net_position}")
                    self.last_execution_ts = state.timestamp
                    self.last_limit_ts = state.timestamp

        # STRATEGY C: Impact Trigger
        # We can check market_trades for large volume in the last tick
        big_trade = self.get_big_trade_volume(state.market_trades.get(KELP, []))
        if big_trade >= self.config["large_trade_volume"]:
            # Check sign of OBI & drift
            if np.sign(obi_smooth) == np.sign(tfi_raw) and abs(drift) > (self.config["theta"] or 0.01):
                side = int(np.sign(tfi_raw))
                newpos = self.net_position + side*self.config["execution_unit_size"]
                if abs(newpos) <= self.config["max_position"]:
                    fill_price = best_ask if side>0 else best_bid
                    self.execute_trade(side, self.config["execution_unit_size"], fill_price, orders_to_place)
                    self.last_execution_ts = state.timestamp

        # STRATEGY D: Breakout Approach
        # We don’t have row["rolling_max_5"] or row["rolling_min_5"], but we can track them ourselves:
        # A simple approach: track the max/min of mid_prices over the last 5 ticks:
        if len(self.mid_prices) >= 5:
            rmax_5 = max(list(self.mid_prices)[-5:])
            rmin_5 = min(list(self.mid_prices)[-5:])
            # Up breakout
            if (
                mid > rmax_5 + self.config["breakout_threshold_up"]
                and drift > (self.config["theta"] or 0.01)
                and obi_smooth > 0
                and tfi_raw > 0
            ):
                side = +1
                canpos = self.net_position + side*self.config["execution_unit_size"]
                if abs(canpos) <= self.config["max_position"]:
                    self.execute_trade(side, self.config["execution_unit_size"], best_ask, orders_to_place)
                    self.last_execution_ts = state.timestamp
            # Down breakout
            elif (
                mid < rmin_5 - self.config["breakout_threshold_down"]
                and drift < -(self.config["theta"] or 0.01)
                and obi_smooth < 0
                and tfi_raw < 0
            ):
                side = -1
                canpos = self.net_position + side*self.config["execution_unit_size"]
                if abs(canpos) <= self.config["max_position"]:
                    self.execute_trade(side, self.config["execution_unit_size"], best_bid, orders_to_place)
                    self.last_execution_ts = state.timestamp

        # 5) Check if signal flips
        self.check_signal_flip(drift, best_bid, best_ask, state.timestamp, orders_to_place)

        return {KELP: orders_to_place}, 1, None

    # ------------------------------------------------------------------------
    # Helper: “execute_trade” simply appends an Order for that side & size
    # ------------------------------------------------------------------------
    def execute_trade(self, side: int, size: int, price: float, orders_to_place: list):
        """
        In a real environment, you don’t forcibly fill your own orders. Instead
        you place an order crossing the spread so it’s extremely likely to fill
        immediately. This replicates your old “execute_trade(...)” approach.
        """
        old_pos = self.net_position
        new_pos = old_pos + side*size
        if old_pos == 0 and new_pos != 0:
            self.position_entry_ts = 0  # We'll set it later to the current timestamp from run
            self.best_price_in_position = price
        elif new_pos == 0:
            self.position_entry_ts = None
            self.best_price_in_position = None

        # Actually place the order
        orders_to_place.append(Order(KELP, int(price), side*size))
        # We'll update self.net_position now, to keep internal state in sync:
        self.net_position = new_pos

        # Just a debug print
        print(f"[TRADE] side={side}, size={size}, price={price:.2f}, newPos={self.net_position}")

    # ------------------------------------------------------------------------
    # Helper: get best_bid & best_ask from the order depth
    # ------------------------------------------------------------------------
    def get_best_bid_ask(self, order_depth: OrderDepth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None, None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return best_bid, best_ask

    # ------------------------------------------------------------------------
    # Helper: OBI
    # ------------------------------------------------------------------------
    def compute_obi(self, order_depth: OrderDepth) -> float:
        total_buys = sum(abs(v) for v in order_depth.buy_orders.values())
        total_sells = sum(abs(v) for v in order_depth.sell_orders.values())
        denominator = total_buys + total_sells or 1e-6
        return (total_buys - total_sells)/denominator

    # ------------------------------------------------------------------------
    # Helper: TFI from market trades
    # ------------------------------------------------------------------------
    def compute_tfi(self, recent_trades: list[Trade], current_ts: int) -> float:
        """
        A basic approach: if the last trade(s) were at an up-tick price, treat direction=+1,
        down-tick => -1, sum up quantities accordingly. This is only a rough approach.
        You can refine it to match your offline logic more closely.
        """
        if not recent_trades:
            return 0.0
        # Example: look at just the last trade
        last_trade = recent_trades[-1]
        # We check if price is up or down from the second-to-last trade
        if len(recent_trades) < 2:
            return 0.0

        second_last_trade = recent_trades[-2]
        direction = 1 if (last_trade.price > second_last_trade.price) else \
                     (-1 if (last_trade.price < second_last_trade.price) else 0)
        total_vol = last_trade.quantity
        return direction*total_vol

    # ------------------------------------------------------------------------
    # Helper: drift from mid_prices
    # ------------------------------------------------------------------------
    def compute_drift(self, window=10) -> float:
        """
        Approximate your old “expected_drift.” We do a simple approach:
          drift = (mid[-1] - mid[-(window+1)]) / mid[-(window+1)]
        You can refine or store any future-based logic if you want.
        """
        if len(self.mid_prices) <= window:
            return 0.0
        old_price = list(self.mid_prices)[-window-1]
        new_price = list(self.mid_prices)[-1]
        if old_price == 0:
            return 0.0
        return (new_price - old_price)/abs(old_price)

    def update_theta(self, drift: float):
        """
        In your original code, “theta” was set to 0.5 * max(abs(drift)).
        But we don’t have a big dataset. We can at least do:
          self.config["theta"] = max(self.config["theta"], abs(drift)*0.5, 0.1)
        or something similar. Adjust as you see fit.
        """
        old_theta = self.config["theta"]
        new_theta = max(0.1, abs(drift)*0.5, old_theta)
        self.config["theta"] = new_theta

    # ------------------------------------------------------------------------
    # Strategy C check: Large trade volume
    # ------------------------------------------------------------------------
    def get_big_trade_volume(self, trades: list[Trade]) -> int:
        """
        Return the biggest volume found in trades. If empty, returns 0.
        """
        if not trades:
            return 0
        max_vol = max(abs(t.quantity) for t in trades)
        return max_vol

    # ------------------------------------------------------------------------
    # Checking trailing stops
    # ------------------------------------------------------------------------
    def check_trailing_stop(self, mid: float, best_bid: float, best_ask: float, ts: int):
        if self.net_position == 0:
            self.best_price_in_position = None
            return

        side_held = int(math.copysign(1, self.net_position))
        if self.best_price_in_position is None:
            self.best_price_in_position = mid
            return

        stop_dist = self.config["trailing_stop_distance"]
        # If we’re long:
        if side_held > 0:
            if mid > self.best_price_in_position:
                self.best_price_in_position = mid
            if (self.best_price_in_position - mid) >= stop_dist:
                # Flatten
                flatten_side = -side_held
                size_to_exit = abs(self.net_position)
                fill_price = best_bid  # sell at bid
                self.execute_trade(flatten_side, size_to_exit, fill_price, [])
                self.last_execution_ts = ts

        # If we’re short:
        elif side_held < 0:
            if mid < self.best_price_in_position:
                self.best_price_in_position = mid
            if (mid - self.best_price_in_position) >= stop_dist:
                # Flatten
                flatten_side = -side_held
                size_to_exit = abs(self.net_position)
                fill_price = best_ask  # cover at ask
                self.execute_trade(flatten_side, size_to_exit, fill_price, [])
                self.last_execution_ts = ts

    # ------------------------------------------------------------------------
    # Checking time-based exit
    # (In a real environment, we’d track self.position_entry_ts with the actual TS
    #  whenever we go from 0 to a nonzero position.)
    # ------------------------------------------------------------------------
    def check_time_exit(self, current_ts: int, drift: float, best_bid: float, best_ask: float):
        if self.net_position == 0 or self.position_entry_ts is None:
            return
        # Suppose we started the position at self.last_execution_ts
        # The old code checked if (ts_now - self.position_entry_ts) > 2000, etc.
        hold_limit = 2000
        if (current_ts - self.last_execution_ts) > hold_limit:
            side_signal = int(math.copysign(1, drift)) if abs(drift) > (self.config["theta"] or 0.01) else 0
            side_held = int(math.copysign(1, self.net_position))
            if side_signal != 0 and side_signal != side_held:
                # Flatten
                flatten_side = -side_held
                size_to_exit = abs(self.net_position)
                fill_price = best_ask if flatten_side>0 else best_bid
                self.execute_trade(flatten_side, size_to_exit, fill_price, [])
                print(f"[TIME STOP EXIT] Flatten side={flatten_side}, pos={size_to_exit} at {fill_price}")

    # ------------------------------------------------------------------------
    # Checking signal flips
    # ------------------------------------------------------------------------
    def check_signal_flip(self, drift: float, best_bid: float, best_ask: float, current_ts: int, orders_to_place: list):
        if self.net_position == 0:
            return
        side_signal = int(math.copysign(1, drift)) if abs(drift) > (self.config["theta"] or 0.01) else 0
        side_held = int(math.copysign(1, self.net_position))
        if side_signal != 0 and side_signal != side_held:
            # Flatten
            flatten_side = -side_held
            size_to_exit = abs(self.net_position)
            fill_price = best_ask if flatten_side>0 else best_bid
            self.execute_trade(flatten_side, size_to_exit, fill_price, orders_to_place)
            print(f"[SIGNAL FLIP EXIT] Flatten side={flatten_side}, pos={size_to_exit} at {fill_price}")
            self.last_execution_ts = current_ts

    # ------------------------------------------------------------------------
    # Estimation of fill probability for passive orders
    # ------------------------------------------------------------------------
    def estimate_fill_probability(self, spread: float, obi_smooth: float) -> float:
        if spread == 2:
            return 0.6 if abs(obi_smooth) > 0.005 else 0.3
        elif spread == 3:
            return 0.3 if abs(obi_smooth) > 0.005 else 0.1
        return 0.0

