"""
WORLD CUP 2026 WINNER PREDICTION — full pipeline in one file
Run:  python wc2026.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from itertools import combinations

DATA = '.'

# ---------- Steps 1-2: train the match model ----------
hist = pd.read_csv(f'{DATA}/wc_matches_historical.csv')
hist['elo_diff'] = hist.home_pre_match_elo - hist.away_pre_match_elo
hist['outcome'] = hist.result_type.map(
    {'Away Win': 0, 'Draw': 1, 'Home Win': 2})
model = LogisticRegression(max_iter=1000).fit(hist[['elo_diff']], hist.outcome)

# ---------- Step 3: symmetric match probabilities ----------


def match_probs(elo_a, elo_b):
    d = elo_a - elo_b
    f = model.predict_proba([[d]])[0]
    r = model.predict_proba([[-d]])[0]
    return np.array([(f[2]+r[0])/2, (f[1]+r[1])/2, (f[0]+r[2])/2])


feats = pd.read_csv(f'{DATA}/wc_prediction_features_2026.csv')
ELO = dict(zip(feats.team, feats.elo_rating_2026))
PROBS = {(a, b): match_probs(ELO[a], ELO[b])
         for a in ELO for b in ELO if a != b}

groups_df = pd.read_csv(f'{DATA}/wc_2026_groups.csv')
GROUPS = {g: list(d.team) for g, d in groups_df.groupby('group')}
rng = np.random.default_rng(42)

# ---------- Step 4: tournament simulation ----------


def play_match(a, b):
    return rng.choice(3, p=PROBS[(a, b)])


def play_knockout(a, b):
    r = play_match(a, b)
    if r == 1:
        p_a = 0.5 + (ELO[a] - ELO[b]) / 4000
        return a if rng.random() < p_a else b
    return a if r == 0 else b


def simulate_group(teams):
    pts = {t: 0 for t in teams}
    for a, b in combinations(teams, 2):
        r = play_match(a, b)
        if r == 0:
            pts[a] += 3
        elif r == 2:
            pts[b] += 3
        else:
            pts[a] += 1
            pts[b] += 1
    order = sorted(teams, key=lambda t: (pts[t], rng.random()), reverse=True)
    return order, pts


def simulate_tournament():
    winners, runners, thirds = [], [], []
    for g, teams in GROUPS.items():
        order, pts = simulate_group(teams)
        winners.append(order[0])
        runners.append(order[1])
        thirds.append((order[2], pts[order[2]]))
    thirds.sort(key=lambda x: (x[1], rng.random()), reverse=True)
    best_thirds = [t for t, _ in thirds[:8]]

    pot_low = runners + best_thirds
    rng.shuffle(winners)
    rng.shuffle(pot_low)
    round32 = [(w, pot_low[i]) for i, w in enumerate(winners)]
    leftovers = pot_low[12:]
    round32 += [(leftovers[i], leftovers[i+1]) for i in range(0, 8, 2)]

    alive = [play_knockout(a, b) for a, b in round32]
    while len(alive) > 1:
        rng.shuffle(alive)
        alive = [play_knockout(alive[i], alive[i+1])
                 for i in range(0, len(alive), 2)]
    return alive[0]


# ---------- Step 5: run simulations and print ----------
N = 10_000
titles = defaultdict(int)
for i in range(N):
    titles[simulate_tournament()] += 1

result = (pd.Series(titles, name='titles') /
          N * 100).sort_values(ascending=False)
print(f"Title probability over {N:,} simulated tournaments:\n")
for team, pct in result.head(15).items():
    print(f"  {team:<15} {pct:5.1f}%  {'#' * int(pct)}")

# ---------- Step 6: make the chart ----------
top = result.head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(top.index, top.values, color='#2e7d32')
bars[-1].set_color('#f9a825')
for bar, val in zip(bars, top.values):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=10)
ax.set_xlabel('Probability of winning the title (%)')
ax.set_title(
    'World Cup 2026 — Title Probability\n(10,000 Monte Carlo simulations)')
ax.set_xlim(0, top.values.max() * 1.15)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('wc2026_chart.png', dpi=150)
plt.show()
