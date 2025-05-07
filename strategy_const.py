import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict
plt.rcParams["figure.dpi"] = 120

# ---------- 1. Inputs ----------
goods  = ["Cacti Needle","Solar panels","Quantum Coffee",
          "Red Flags","VR Monocle",
          "Ranch Sauce","Striped shirts","Haystacks","Moonshine"]
scores = np.array([-3,-3,-3, 3,3, 2,2, 1,1])

pars = { 3:dict(p=.8,up=.10,down=-.05,sigma=.10),
         2:dict(p=.7,up=.07,down=-.04,sigma=.07),
         1:dict(p=.6,up=.04,down=-.03,sigma=.05),
        -1:dict(p=.6,up=-.04,down=.03,sigma=.05),
        -2:dict(p=.7,up=-.07,down=.04,sigma=.07),
        -3:dict(p=.8,up=-.10,down=.05,sigma=.10)}

rows=[]
for g,s in zip(goods,scores):
    p,u,d,σ = (pars[s][k] for k in ("p","up","down","sigma"))
    rows.append(dict(Good=g,Score=s,E_ret=p*u+(1-p)*d,Sigma=σ))
news = pd.DataFrame(rows)

# ---------- 2. Optimiser ----------
def optimise(E, σ, α, cap=1.00):
    """return fractional weights (long=+, short=-)"""
    kelly = 0.25*E/σ**2
    w = np.sign(kelly) * np.maximum(np.abs(kelly) - α/(4*σ**2), 0.)
    tot = np.abs(w).sum()
    if tot > cap and tot > 0:
        w *= cap/tot                              # scale to cap
    # integerise to %-points
    pct = np.rint(100*w).astype(int)
    diff = 100 - np.abs(pct).sum()
    while diff != 0:
        j = np.argmax(np.abs(w))                  # adjust biggest
        pct[j] += np.sign(diff)
        diff = 100 - np.abs(pct).sum()
    return pct/100.0                              # back to fractions

# ---------- 3. Monte‑Carlo with fee charged ----------
@dataclass
class Simulator:
    df : pd.DataFrame
    prs: Dict[int,Dict]
    def run(self, w_frac, N=30000, seed=0):
        cap = 1000000
        rng = np.random.default_rng(seed)
        pnl = np.zeros(N)
        for i,row in self.df.iterrows():
            if w_frac[i]==0: continue
            p,u,d = (self.prs[row.Score][k] for k in ("p","up","down"))
            ret   = np.where(rng.random(N)<p, u, d)
            pnl  += cap*w_frac[i]*ret            # trading P&L
            pnl  -= α_grid_current*cap*abs(w_frac[i])  # fee in cash
        return pnl

@dataclass
class Plotter:
    sc : List[Dict]; names: List[str]
    def draw(self):
        plt.figure(figsize=(16,12))
        for k,s in enumerate(self.sc,1):
            ax = plt.subplot(4,2,k)
            ax.hist(s["pnl"], bins=60)
            ax.axvline(0,color='k',ls='--')
            ax.set_title(s["lbl"])
            if k>=7: ax.set_xlabel("P&L (SS)")
            if k in (1,3,5,7): ax.set_ylabel("Freq")
            txt=[f"{n[:10]:11s} {'B' if p>0 else 'S'} {abs(p):2d}%"
                 for n,p in zip(self.names, s["pct"])]
            ax.text(0.02,0.98,"\n".join(txt),va='top',ha='left',
                    transform=ax.transAxes,fontsize=7,family='monospace',
                    bbox=dict(fc='white',alpha=.7,boxstyle='round'))
        plt.suptitle("P&L distributions – fee grid α charged in cash",fontsize=14)
        plt.tight_layout(rect=[0,0.03,1,0.95]); plt.show()

# ---------- 4. Fee sweep ----------
alpha_grid = [0.005,0.02,0.05,0.08,0.10,0.20,0.50,0.80]
sim   = Simulator(news, pars)
runs  = []

for seed,α in enumerate(alpha_grid,1):
    α_grid_current = α                                   # for simulator fee line
    w   = optimise(news.E_ret.values, news.Sigma.values, α)
    pnl = sim.run(w, seed=seed)
    runs.append(dict(lbl=f"α={α:.1%}",
                     pnl=pnl,
                     pct=np.round(100*w).astype(int),
                     w=w,
                     mean = pnl.mean(),
                     p95  = np.percentile(pnl,95)))

# ---------- 5. Plot & summary ----------
Plotter(runs, goods).draw()

summary = pd.DataFrame({
    "Scenario" : [r["lbl"] for r in runs],
    "Mean P&L" : [f"{r['mean']:,.0f}" for r in runs],
    "95‑th pct": [f"{r['p95']:,.0f}"  for r in runs]
})
print("\nAll‑scenario P&L summary")
print(summary)

best = max(runs, key=lambda d:d["mean"])
best_tbl = (pd.DataFrame({"Good":goods,
                          "Direction":np.where(best["pct"]>0,"BUY","SELL"),
                          "WeightPct":np.abs(best["pct"])})
            .query("WeightPct>0")
            .sort_values("WeightPct",ascending=False)
            .reset_index(drop=True))

print(f"\nBest scenario → {best['lbl']}"
      f"\nMean P&L  = {best['mean']:,.0f} SS"
      f"\n95‑th pct = {best['p95']:,.0f} SS")
best_tbl
