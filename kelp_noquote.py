import math
import numpy as np
from datamodel import Order, Trade, OrderDepth, TradingState

KELP = "KELP"
PRODUCTS = [KELP]

MAX_POSITION = 50

class Trader:

    def __init__(self):

        # For the OU mean-reversion model, store rolling prices for KELP
        self.prices_history = []
        # Keep track of the current OU parameters
        self.mu = 0.0
        self.theta = 0.005
        self.sigma = 0.0
        # Rolling window to estimate variance
        self.rolling_window = 50

        # For a simple “trailing average” approach if needed
        self.simple_ma_window = 40
        self.prices_for_sma = []

        # Keep track of last mid price for returns
        self.last_mid_price = None

        # Book-keeping for last time we updated OU parameters
        self.last_param_update_timestamp = 0
        self.param_update_interval = 2000  # update OU params every 2000ms (for example)

        print("Initialized Mixed OU + Market Making Trader")

    def compute_mid_price(self, state: TradingState) -> float:
        """
        Simple best-bid / best-ask mid.  Fallback if data missing, return some default like 2030.
        """
        order_depth = state.order_depths.get(KELP, None)
        if not order_depth or (len(order_depth.buy_orders) == 0) or (len(order_depth.sell_orders) == 0):
            return 2030.0  # fallback
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid = 0.5 * (best_bid + best_ask)
        return mid

    def fit_ou_params(self):
        """
        Basic rough fit of OU parameters using consecutive returns from the stored price history.
        For example, estimate drift as average daily difference vs. a mean, etc.
        This is *very rough*; you can do more robust stats if you’d like.
        """
        # We need enough data
        if len(self.prices_history) < 2:
            return

        # Convert price series to *differences* or returns
        diffs = []
        for i in range(1, len(self.prices_history)):
            diff = self.prices_history[i] - self.prices_history[i - 1]
            diffs.append(diff)
        if len(diffs) < 2:
            return

        # Simple approach: estimate sample mean reversion by correlating diffs with level
        # E.g. dX = theta*(mu - X) dt + sigma dW
        # We'll do a quick method-of-moments approach:
        avg_price = np.mean(self.prices_history)
        # Crude guess for theta: if diffs are on average negative when price > average, etc.
        # This is *extremely simplistic.*  You can do better with standard OU MLE.
        # We'll do a correlation approach: correlation( X_t - mu, dX_t ) ~ -theta * var(X).
        X = np.array(self.prices_history[:-1])
        Y = np.array(diffs)
        # center X about the sample mean
        Xc = X - avg_price

        numerator = np.sum(Xc * Y)
        denominator = np.sum(Xc * Xc) + 1e-9
        # correlation
        raw_corr = numerator / denominator  # should be roughly -theta
        # so:
        self.theta = max(0.001, -raw_corr)  # keep it positive but small
        # estimated mu
        self.mu = avg_price

        # for sigma, quick stdev of diffs
        stdev_diff = np.std(diffs)
        self.sigma = stdev_diff * math.sqrt(self.theta)  # rough
        # can tweak this formula to your liking

    def run(self, state: TradingState):
        """
        Return the dictionary: symbol -> list of Orders
        We'll do a simple combined approach:
          1) directional: if current price is above OU "fair", we place a small sell
             if below, we place a small buy
          2) market-making: place small symmetrical limit orders around mid price
        """
        orders_by_symbol = {}

        # 1) get mid
        mid = self.compute_mid_price(state)

        # update rolling price arrays
        self.prices_history.append(mid)
        if len(self.prices_history) > 2000:
            self.prices_history.pop(0)

        # maybe update OU parameters every so often
        if (state.timestamp - self.last_param_update_timestamp) >= self.param_update_interval:
            self.fit_ou_params()
            self.last_param_update_timestamp = state.timestamp
            print(f"[DEBUG] OU params updated: mu={self.mu:.2f}, theta={self.theta:.4f}, sigma={self.sigma:.4f}")

        # 2) directional signal from OU
        # If price > mu, we lean short. If price < mu, we lean long.
        # Weighted by how far from mu in stdev units?
        dist = mid - self.mu
        stdev_est = max(1.0, self.sigma)  # guard from zero
        zscore = dist / (stdev_est)  # rough
        # e.g. if zscore > 1, we want to short more
        # Also watch position limits
        position = state.position.get(KELP, 0)

        # We'll define a small target "size" based on zscore
        # for example, position_target = -zscore * 10  (10 shares per zscore)
        # Then place orders that move us from current to that target
        position_target = -zscore * 10.0

        # clamp target to [-MAX_POSITION, +MAX_POSITION]
        position_target = max(-MAX_POSITION, min(MAX_POSITION, position_target))

        # desired trade = target - current
        trade_needed = position_target - position
        # round to an integer
        trade_needed = int(round(trade_needed))

        # We'll place a single limit order to try to accomplish part/all of that.
        # If trade_needed > 0 => buy
        # If trade_needed < 0 => sell
        # Because orders can’t be canceled, we’ll place them near the current mid or slightly better
        # to have a chance to fill, but not too aggressively.

        ou_orders = []
        if trade_needed > 0:
            # place a buy
            buy_price = int(math.floor(mid - 1))  # 1 tick below mid
            # clamp so we don’t exceed the leftover position
            qty = min(trade_needed, MAX_POSITION - position)
            if qty > 0:
                ou_orders.append(Order(KELP, buy_price, qty))
        elif trade_needed < 0:
            # place a sell
            sell_price = int(math.ceil(mid + 1))  # 1 tick above mid
            qty = max(trade_needed, -MAX_POSITION - position)
            if qty < 0:
                ou_orders.append(Order(KELP, sell_price, qty))

        # 3) simple market-making layer: always place a small buy limit 2 ticks below mid
        #    and a small sell limit 2 ticks above mid, if it doesn’t exceed max position.
        mm_orders = []

        # figure out how many we can still buy or sell
        can_buy = MAX_POSITION - position
        can_sell = position + MAX_POSITION  # if position is negative, we have more room to sell

        # place small symmetrical orders, e.g. size = 2
        # buy:
        if can_buy > 0:
            mm_bid_price = int(math.floor(mid - 2))
            mm_bid_qty = 2 if can_buy >= 2 else can_buy
            if mm_bid_qty > 0:
                mm_orders.append(Order(KELP, mm_bid_price, mm_bid_qty))

        # sell:
        if can_sell > 0:
            mm_ask_price = int(math.ceil(mid + 2))
            mm_ask_qty = 2 if can_sell >= 2 else can_sell
            # but we want negative quantity to sell
            mm_ask_qty = -mm_ask_qty
            if mm_ask_qty < 0:
                mm_orders.append(Order(KELP, mm_ask_price, mm_ask_qty))

        # combine
        all_orders = ou_orders + mm_orders
        orders_by_symbol[KELP] = all_orders

        return orders_by_symbol
