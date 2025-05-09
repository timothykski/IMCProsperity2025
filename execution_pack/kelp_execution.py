import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from config import StrategyConfig
from trade_record import TradeRecord



class TradeKELP:
    def __init__(self, prices_df: pd.DataFrame, trades_df: pd.DataFrame, config: StrategyConfig):
        self.prices = prices_df.copy().reset_index(drop=True)
        self.trades = trades_df.copy().reset_index(drop=True)
        self.config = config

        # We'll store final executed trades here for PnL
        self.execution_log = []
        # We track open positions from Strategy A
        self.open_positions = []
        # We track passive limit orders
        self.limit_orders = []

        self.last_execution_ts = -np.inf

    def compute_mid_price(self):
        self.prices['mid_price'] = (self.prices['bid_price_1'] + self.prices['ask_price_1']) / 2

    def compute_spread(self):
        self.prices['spread'] = self.prices['ask_price_1'] - self.prices['bid_price_1']

    def compute_obi(self):
        # Rolling logic
        bid_vol = sum(self.prices.get(f'bid_volume_{i}', 0).fillna(0) for i in range(1, 4))
        ask_vol = sum(self.prices.get(f'ask_volume_{i}', 0).fillna(0) for i in range(1, 4))
        self.prices['obi'] = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-6)
        self.prices['obi_smooth'] = self.prices['obi'].rolling(5,min_periods=1).mean()

    def compute_tfi(self):
        self.trades['timestamp_bin'] = (self.trades['timestamp'] // 1000) * 1000
        self.trades['direction'] = np.where(
            self.trades['price'] > self.trades['price'].shift(1), 1,
            np.where(self.trades['price'] < self.trades['price'].shift(1), -1, 0)
        )
        self.trades['signed_qty'] = self.trades['direction'] * self.trades['quantity']
        tfi = self.trades.groupby('timestamp_bin')['signed_qty'].sum().reset_index(name='tfi')
        self.prices['timestamp_bin'] = (self.prices['timestamp'] // 1000) * 1000
        self.prices = self.prices.merge(tfi, on='timestamp_bin', how='left').fillna({'tfi': 0})

    def compute_expected_drift(self):
        self.prices['future_mid'] = self.prices['mid_price'].shift(-10)
        self.prices['expected_drift'] = self.prices['future_mid'] - self.prices['mid_price']
        self.prices['expected_drift'] = self.prices['expected_drift'].fillna(0)
        self.set_dynamic_theta()

    def set_dynamic_theta(self):
        max_drift = self.prices['expected_drift'].abs().max()
        self.config.theta = 0.5 * max_drift if max_drift > 0 else 0.1
        print(f"Dynamic theta set to: {self.config.theta:.4f}")

    def get_position_sum(self):
        """Number of total open positions from Strategy A + passive filled orders (not closed)."""
        pos_sum = 0
        for p in self.open_positions:
            pos_sum += p.side * p.size
        for l in self.limit_orders:
            if l.filled and l.close_price is None:  # means limit got filled, not closed
                pos_sum += l.side * l.size
        return pos_sum

    def act_on_signals(self):
        last_limit_ts = -np.inf

        for idx, row in self.prices.iterrows():
            spread = row['spread']
            obi_smooth = row['obi_smooth']
            tfi = row['tfi']
            drift = row['expected_drift']
            ts = row['timestamp']
            mid_price = row['mid_price']

            self.update_passive_orders(ts, mid_price)
            self.close_positions(ts, mid_price)

            # Simple decay check
            if ts - self.last_execution_ts < self.config.signal_decay_ms:
                continue

            # STRATEGY A: AGGRESSIVE
            if spread <= 2 \
               and abs(drift) > self.config.theta \
               and abs(obi_smooth) > self.config.obi_threshold \
               and np.sign(obi_smooth) == np.sign(tfi):

                direction = int(np.sign(drift))
                # check position sum + new trade doesn't exceed 50
                if abs(self.get_position_sum() + direction*self.config.execution_unit_size) <= self.config.max_position:
                    # open an immediate position
                    trade = TradeRecord(
                        ts=ts, side=direction, size=self.config.execution_unit_size,
                        entry_price=mid_price + (0.5 if direction>0 else -0.5),
                        strategy="A",
                    )
                    self.open_positions.append(trade)
                    self.execution_log.append({
                        "timestamp": ts,
                        "type":"market_open",
                        "side": direction,
                        "size": self.config.execution_unit_size,
                        "entry_price": trade.entry_price
                    })
                    self.last_execution_ts = ts
                    print(f"[Strategy A] Opened position side={direction}, size=1 at {trade.entry_price:.2f}")

            # STRATEGY B: PASSIVE
            elif 0.005<=abs(obi_smooth)<=0.01 and (np.sign(obi_smooth) == np.sign(tfi) or abs(tfi) <1e-3):
                if spread in [2,3] and ts - last_limit_ts>= self.config.cooldown_ms:
                    direction = 1 if obi_smooth>0 else -1
                    if abs(self.get_position_sum() + direction*self.config.execution_unit_size)<= self.config.max_position:
                        # place limit
                        limit_price= (mid_price -0.5) if direction<0 else (mid_price +0.5)
                        if spread==3:  # queue jump 1 tick inside
                            limit_price= mid_price+ (1.0 if direction>0 else -1.0)
                        trade = TradeRecord(
                            ts=ts, side=direction, size=self.config.execution_unit_size,
                            entry_price=limit_price,
                            strategy="B",
                            expiry_ts=ts+self.config.limit_ttl_ms
                        )
                        self.limit_orders.append(trade)
                        self.execution_log.append({
                            "timestamp": ts,
                            "type":"limit_placed",
                            "side": direction,
                            "size": self.config.execution_unit_size,
                            "limit_price": limit_price
                        })
                        last_limit_ts= ts
                        self.last_execution_ts= ts
                        print(f"[Strategy B] Placed limit side={direction}, price={limit_price:.2f}")

            # IMPACT TRIGGER
            recent_trade= self.trades[self.trades["timestamp"]<= ts].tail(1)
            if not recent_trade.empty:
                last_vol= recent_trade.iloc[0]["quantity"]
                if last_vol>= self.config.large_trade_volume and np.sign(obi_smooth)==np.sign(tfi) and abs(drift)> self.config.theta:
                    side= int(np.sign(tfi))
                    if abs(self.get_position_sum()+ side*self.config.execution_unit_size)<= self.config.max_position:
                        trade= TradeRecord(
                            ts=ts, side=side, size=self.config.execution_unit_size,
                            entry_price= mid_price+ (0.5 if side>0 else -0.5),
                            strategy="Impact",
                        )
                        self.open_positions.append(trade)
                        self.execution_log.append({
                            "timestamp": ts,
                            "type":"impact_open",
                            "side": side,
                            "size": self.config.execution_unit_size,
                            "entry_price": trade.entry_price
                        })
                        self.last_execution_ts= ts
                        print(f"[Impact] Large vol trade side={side}, opened position")

    def update_passive_orders(self, current_ts, current_mid):
        """Check if any limit orders get filled or expire."""
        for limit in self.limit_orders:
            if limit.filled or limit.close_price is not None:
                continue  # already filled or closed
            # check expiry
            if current_ts >= limit.expiry_ts:
                # expired
                self.execution_log.append({
                    "timestamp": current_ts,
                    "type": "limit_expire",
                    "side": limit.side,
                    "size": limit.size
                })
                limit.close_price= limit.entry_price
                limit.filled= False
                continue
            # check fill if mid crosses limit price
            # if side=1 => limit price> mid => fill
            # if side=-1 => limit price< mid => fill
            if (limit.side>0 and current_mid>= limit.entry_price) \
               or (limit.side<0 and current_mid<= limit.entry_price):
                # got filled
                limit.filled= True
                self.execution_log.append({
                    "timestamp": current_ts,
                    "type":"limit_filled",
                    "side":limit.side,
                    "size":limit.size,
                    "fill_price":limit.entry_price
                })
                print(f"[Strategy B] limit filled side={limit.side}, fill_price={limit.entry_price:.2f} @ {current_ts}")

    def close_positions(self, current_ts, current_mid):
        """Close out any positions that have hit hold_ticks or end the next tick."""
        # close out strategy A or Impact after hold_ticks
        to_close= []
        for pos in self.open_positions:
            idx= self.prices.index[self.prices['timestamp']==pos.ts]
            if len(idx)== 0:
                continue
            # if the current row index> pos_row+ hold_ticks => we forcibly close
            pos_row= idx[0]
            current_row_idx= self.prices.index[self.prices['timestamp']== current_ts]
            if len(current_row_idx)==0:
                continue
            c_idx= current_row_idx[0]
            if c_idx>= pos_row+ self.config.hold_ticks:
                # close
                pos.close_ts= current_ts
                pos.close_price= current_mid
                to_close.append(pos)

        for c in to_close:
            self.open_positions.remove(c)
            realized_pnl= c.side* c.size* (current_mid- c.entry_price)
            self.execution_log.append({
                "timestamp": c.close_ts,
                "type":"position_close",
                "side": c.side,
                "size": c.size,
                "entry_price": c.entry_price,
                "close_price": c.close_price,
                "pnl": realized_pnl
            })
            print(f"Closed pos from {c.strategy} with PnL={realized_pnl:.2f}")

class PnLEvaluator:
    def __init__(self, prices_df: pd.DataFrame, execution_log: list):
        self.prices= prices_df.copy().reset_index(drop=True)
        self.logs= pd.DataFrame(execution_log)

    def compute(self):
        # gather all trades with 'position_close' or 'limit_filled'
        # for limit_filled => we still hold => PnL realized only when we forcibly close or expiry
        closes= self.logs[self.logs['type']=='position_close'].copy()
        if closes.empty:
            # no realized
            cum_pnl= [0]* len(self.logs)
            self.logs['cum_pnl']= cum_pnl
            return self.logs
        # cumsum
        closes['cumulative_pnl']= closes['pnl'].cumsum()
        # merge back
        cum= 0
        cum_list= []
        for i, row in self.logs.iterrows():
            if row['type']=='position_close':
                cum+= row.get('pnl',0)
            cum_list.append(cum)
        self.logs['cum_pnl']= cum_list
        return self.logs

    def plot_cumulative_pnl(self):
        if 'cum_pnl' not in self.logs.columns:
            print("No 'cum_pnl' column. Possibly no closed trades.")
            return
        plt.figure(figsize=(12,6))
        plt.plot(self.logs['timestamp'], self.logs['cum_pnl'], label='Cumulative PnL')
        plt.axhline(0,color='gray',linestyle='--')
        plt.xlabel("Timestamp")
        plt.ylabel("Cumulative PnL")
        plt.title("Strategy Cumulative PnL Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__=="__main__":
    prices= pd.read_excel("C:/Users/aa04947/OneDrive - APG/Desktop/IMCR1/out/kelp_prices_combined.xlsx")
    trades= pd.read_excel("C:/Users/aa04947/OneDrive - APG/Desktop/IMCR1/out/kelp_trades_combined.xlsx")

    cfg= StrategyConfig()
    trader= TradeKELP(prices, trades, cfg)
    exec_log= trader.run()
    if len(exec_log)==0:
        print("No trades executed.")
    evaluator= PnLEvaluator(prices, exec_log)
    results= evaluator.compute()
    evaluator.plot_cumulative_pnl()
    print(results.tail())
