import math
import numpy as np
from datamodel import Order, OrderDepth, Trade, TradingState

KELP = "KELP"
PRODUCTS = [KELP]

class Trader:
    def __init__(self):
        """
        A hybrid directional + light market-making approach for KELP.

        1) If short-term drift is strongly up (drift>+0.1):
           - we buy up to a small net long (e.g. +15).
           - we skip placing short quotes.
        2) If short-term drift < -0.1:
           - we short up to -15.
           - we skip placing new long quotes.
        3) Otherwise, we do a mild layering around mid to collect the spread.
        4) If we accumulate big position anyway, we flatten partially or place quotes
           that help us reduce it.
        """
        self.max_position = 50
        self.pos = 0

        # We'll place up to 2 layers in “neutral mode”
        self.num_layers = 2
        self.base_size = 3

        # directional_position_limit => how big a long or short we hold in momentum mode
        self.directional_position_limit = 15

        # Rolling data for mid / drift
        self.mid_history = []
        self.vol_window = 20
        self.drift_window = 10

        # If we exceed half of max_position, we shift quotes or flatten.
        self.over_limit_buffer = 0.5 * self.max_position

        # threshold for drift to engage directional mode
        self.drift_threshold = 0.1

    def run(self, state: TradingState):
        # Basic checks
        if KELP not in state.order_depths:
            return {KELP: []}, 1, None
        od: OrderDepth = state.order_depths[KELP]
        best_bid, best_ask = self.get_best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return {KELP: []}, 1, None

        self.pos = state.position.get(KELP, 0)

        mid_price = 0.5*(best_bid + best_ask)
        self.mid_history.append(mid_price)
        if len(self.mid_history) > self.vol_window:
            self.mid_history.pop(0)
        drift = self.compute_short_drift(self.drift_window)
        px_std = np.std(self.mid_history) if len(self.mid_history)>1 else 0.0

        # Possibly flatten if we have a huge negative position but the drift is up, or vice versa
        flatten_orders = self.check_contradiction(drift, best_bid, best_ask)
        if flatten_orders:
            # Return flatten orders only
            return {KELP: flatten_orders}, 1, None

        # Decide if we’re in momentum “directional mode” or neutral “market-making mode”
        orders = []
        can_buy  = self.max_position - self.pos
        can_sell = self.max_position + self.pos

        if drift>self.drift_threshold:
            # UPTREND => hold small net long up to +15
            # skip short side
            target_long = self.directional_position_limit
            if self.pos < target_long:
                # place a near-best-ask buy to get quickly filled
                # we won't place any sells
                needed = min(target_long - self.pos, can_buy)
                if needed>0:
                    # we cross the spread: buy at best_ask
                    orders.append(Order(KELP, int(best_ask), needed))
        elif drift< -self.drift_threshold:
            # DOWNTREND => hold net short up to -15
            # skip long side
            target_short = -self.directional_position_limit
            if self.pos > target_short:
                needed = min(self.pos - target_short, can_sell)
                if needed>0:
                    # we cross the spread: sell at best_bid (open short)
                    orders.append(Order(KELP, int(best_bid), -needed))
        else:
            # NEUTRAL => do mild layering around mid
            base_offset = 1.0
            if px_std>1.0:
                base_offset += (px_std -1.0)*0.5

            # We'll place up to 2 layers
            for i in range(self.num_layers):
                offset = base_offset + i
                bid_px = mid_price - offset
                ask_px = mid_price + offset

                # If pos is quite large, skip that side
                if abs(self.pos) > self.over_limit_buffer:
                    if self.pos>0:
                        # skip new asks? Actually skip new LONGS => skip placing bid
                        bid_px=None
                    else:
                        # skip new SHORTS => skip placing ask
                        ask_px=None

                # no cross
                if ask_px and bid_px and (ask_px<=bid_px):
                    continue

                # volumes
                layer_size = self.base_size
                bvol = min(layer_size, max(0, can_buy)) if bid_px else 0
                svol = min(layer_size, max(0, can_sell)) if ask_px else 0

                if bvol>0:
                    orders.append(Order(KELP, int(bid_px), bvol))
                    can_buy-=bvol
                if svol>0:
                    orders.append(Order(KELP, int(ask_px), -svol))
                    can_sell-=svol

                if can_buy<=0 and can_sell<=0:
                    break

        print(f"[Hybrid] pos={self.pos}, drift={drift:.2f}, px_std={px_std:.2f}, #orders={len(orders)}")
        return {KELP: orders}, 1, None

    # ----------------------------------------------------------------------
    def check_contradiction(self, drift: float, best_bid: float, best_ask: float):
        """
        If we hold a large negative position but drift is strongly positive => flatten
        If we hold a large positive pos but drift is strongly negative => flatten
        Flatten partially or fully, depending on how big the mismatch is.
        """
        flatten_orders = []
        # For instance, if pos<-10 and drift>0 => flatten
        # Or if pos>10 and drift<0 => flatten
        # You can define a threshold, or just do sign-based
        mismatch_threshold = 0.05 #todo: try different thresholds ranging from 0.05 to 0.35

        if self.pos<0 and drift> mismatch_threshold:
            # flatten all short
            needed= abs(self.pos)
            flatten_orders.append(Order(KELP,int(best_ask), needed))
            new_pos= self.pos+ needed
            print(f"[FLATTEN SHORT] from {self.pos} to {new_pos}")
            self.pos=new_pos

        elif self.pos>0 and drift< -mismatch_threshold:
            needed= abs(self.pos)
            flatten_orders.append(Order(KELP,int(best_bid), -needed))
            new_pos= self.pos- needed
            print(f"[FLATTEN LONG] from {self.pos} to {new_pos}")
            self.pos=new_pos

        return flatten_orders

    def get_best_bid_ask(self, od: OrderDepth):
        if not od.buy_orders or not od.sell_orders:
            return None, None
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        return best_bid, best_ask

    def compute_short_drift(self, w: int):
        if len(self.mid_history)<= w:
            return 0.0
        old_px= self.mid_history[-(w+1)]
        new_px= self.mid_history[-1]
        return (new_px - old_px)/(w or 1)
