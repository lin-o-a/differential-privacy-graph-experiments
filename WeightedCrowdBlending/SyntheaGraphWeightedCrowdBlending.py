"""
Synthea — 5-Stratum Pipeline
Metrics: Absolute + Relative (%) + KL Divergence (Cover)

S1-S2: Standard DP only  (already blended naturally)
S3-S5: Weighted crowd blending (outliers need artificial crowd)
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import pickle
from scipy import stats
from scipy.special import rel_entr   # KL divergence

np.random.seed(42)
SAVE_PATH = r'C:\Users\Public\Synthea\\'

# ─────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────
def load_synthea():
    with open(SAVE_PATH + 'graph.pkl', 'rb') as f:
        G = pickle.load(f)
    nodes = pd.read_csv(SAVE_PATH + 'nodes.csv')

    features = {}
    for _, row in nodes.iterrows():
        features[row['Id']] = {
            'condition':  row['condition_count'],
            'medication': row['medication_count'],
            'encounter':  row['encounter_count']
        }

    nodes['z_condition']  = stats.zscore(nodes['condition_count'])
    nodes['z_medication'] = stats.zscore(nodes['medication_count'])
    nodes['z_encounter']  = stats.zscore(nodes['encounter_count'])
    nodes['composite']    = (0.4 * nodes['z_condition'] +
                             0.3 * nodes['z_encounter'] +
                             0.3 * nodes['z_medication'])

    composite = {row['Id']: row['composite'] for _, row in nodes.iterrows()}

    # Feature ranges for relative error
    ranges = {
        'condition':  nodes['condition_count'].max() - nodes['condition_count'].min(),
        'medication': nodes['medication_count'].max() - nodes['medication_count'].min(),
    }

    print(f"✅ Loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"   condition range:  [0, {nodes['condition_count'].max():.0f}]")
    print(f"   medication range: [0, {nodes['medication_count'].max():.0f}]")
    return G, features, composite, nodes, ranges


# ─────────────────────────────────────────
# KL DIVERGENCE — Cover & Thomas
# Measures: how different is synthetic from real?
# D_KL = 0 → identical, higher → more different
# ─────────────────────────────────────────
def kl_divergence(real_vals, synth_vals, n_bins=30):
    """
    KL divergence D_KL(real || synthetic)
    Lower = synthetic distribution closer to real
    """
    all_vals = np.concatenate([real_vals, synth_vals])
    vmin, vmax = all_vals.min(), all_vals.max()
    if vmax == vmin:
        return 0.0

    bins = np.linspace(vmin, vmax, n_bins + 1)

    p, _ = np.histogram(real_vals,  bins=bins, density=True)
    q, _ = np.histogram(synth_vals, bins=bins, density=True)

    # Smooth to avoid log(0)
    eps = 1e-10
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()

    kl = float(np.sum(rel_entr(p, q)))
    return kl


def entropy(vals, n_bins=30):
    """Shannon entropy H(X) — how spread is the distribution?"""
    counts, _ = np.histogram(vals, bins=n_bins)
    counts = counts + 1e-10
    probs  = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-10)))


# ─────────────────────────────────────────
# STRATA
# ─────────────────────────────────────────
def define_five_strata(composite):
    scores   = np.array(list(composite.values()))
    node_ids = list(composite.keys())

    Q1, Q3    = np.percentile(scores, 25), np.percentile(scores, 75)
    iqr_upper = Q3 + 1.5 * (Q3 - Q1)

    p50 = np.percentile(scores, 50)
    p90 = np.percentile(scores, 90)
    p97 = np.percentile(scores, 97)
    p99 = np.percentile(scores, 99)

    strata = {'S1': [], 'S2': [], 'S3': [], 'S4': [], 'S5': []}
    for n in node_ids:
        s = composite[n]
        if   s <= p50:  strata['S1'].append(n)
        elif s <= p90:  strata['S2'].append(n)
        elif s <= p97:  strata['S3'].append(n)
        elif s <= p99:  strata['S4'].append(n)
        else:           strata['S5'].append(n)

    treatments = {
        'S1': ('dp_only',          1.0,   'Standard DP only,     ε=1.0'),
        'S2': ('dp_only',          0.5,   'Standard DP only,     ε=0.5'),
        'S3': ('weighted_medium',  0.2,   'Weighted crowd blend, ε=0.2'),
        'S4': ('weighted_heavy',   0.05,  'Heavy weighted crowd, ε=0.05'),
        'S5': ('weighted_maximum', 0.01,  'Maximum crowd blend,  ε=0.01'),
    }

    iqr_flags = set(n for n in node_ids if composite[n] > iqr_upper)

    print(f"\n{'='*65}")
    print("5-STRATUM BREAKDOWN")
    print(f"{'='*65}")
    for name, nodes_list in strata.items():
        if not nodes_list:
            continue
        iqr_agree = len(set(nodes_list) & iqr_flags)
        treat, eps, desc = treatments[name]
        tag = "← CROWD BLENDING" if treat != 'dp_only' else "← standard DP"
        print(f"  {name}: {len(nodes_list):>5,} patients  "
              f"IQR-agree={100*iqr_agree/max(len(nodes_list),1):.0f}%  "
              f"ε={eps}  {tag}")

    return strata, treatments, iqr_upper


# ─────────────────────────────────────────
# STANDARD DP (S1, S2)
# ─────────────────────────────────────────
def apply_standard_dp(node_list, features, epsilon, ranges):
    noisy = {}
    for n in node_list:
        c = features[n]['condition']
        m = features[n]['medication']
        noisy[n] = {
            'condition':  max(0, c + np.random.laplace(0, 1.0 / epsilon)),
            'medication': max(0, m + np.random.laplace(0, 1.0 / epsilon)),
        }

    # Absolute MAE
    mae_c = np.mean([abs(noisy[n]['condition']  - features[n]['condition'])  for n in node_list])
    mae_m = np.mean([abs(noisy[n]['medication'] - features[n]['medication']) for n in node_list])
    abs_mae = (mae_c + mae_m) / 2

    # Relative MAE (%)
    rel_mae_c = 100 * mae_c / max(ranges['condition'],  1)
    rel_mae_m = 100 * mae_m / max(ranges['medication'], 1)
    rel_mae   = (rel_mae_c + rel_mae_m) / 2

    # Noise error (absolute)
    noise_abs = np.mean([
        abs(noisy[n]['condition']  - features[n]['condition']) +
        abs(noisy[n]['medication'] - features[n]['medication'])
        for n in node_list
    ])
    noise_rel = 100 * noise_abs / (ranges['condition'] + ranges['medication'])

    # KL divergence: noisy vs real
    real_c  = np.array([features[n]['condition']  for n in node_list])
    real_m  = np.array([features[n]['medication'] for n in node_list])
    noisy_c = np.array([noisy[n]['condition']  for n in node_list])
    noisy_m = np.array([noisy[n]['medication'] for n in node_list])
    kl_c = kl_divergence(real_c, noisy_c)
    kl_m = kl_divergence(real_m, noisy_m)

    # Shannon entropy of real features
    h_c = entropy(real_c)
    h_m = entropy(real_m)

    return noisy, abs_mae, rel_mae, noise_abs, noise_rel, kl_c, kl_m, h_c, h_m


# ─────────────────────────────────────────
# DP STATS RELEASE
# ─────────────────────────────────────────
def dp_stats(node_list, features, epsilon):
    c    = np.array([features[n]['condition']  for n in node_list])
    m    = np.array([features[n]['medication'] for n in node_list])
    sens = 1.0 / len(node_list)
    c_mean = float(np.clip(np.mean(c) + np.random.laplace(0, sens/epsilon), 0, None))
    c_std  = float(max(0.01, np.std(c) + np.random.laplace(0, sens/epsilon)))
    m_mean = float(np.clip(np.mean(m) + np.random.laplace(0, sens/epsilon), 0, None))
    m_std  = float(max(0.01, np.std(m) + np.random.laplace(0, sens/epsilon)))
    return c_mean, c_std, m_mean, m_std


# ─────────────────────────────────────────
# WEIGHTED CROWD (S3, S4, S5)
# ─────────────────────────────────────────
def weighted_crowd(s1_nodes, features,
                   tc_mean, tc_std, tm_mean, tm_std,
                   n_synth, epsilon, s1_fraction):
    s1_cm, s1_cs, s1_mm, s1_ms = dp_stats(s1_nodes, features, epsilon)

    n_s1  = int(n_synth * s1_fraction)
    n_tgt = n_synth - n_s1

    cond = np.concatenate([
        np.clip(np.random.normal(s1_cm,   s1_cs,   n_s1),  0, None),
        np.clip(np.random.normal(tc_mean, tc_std,  n_tgt), 0, None)
    ])
    med = np.concatenate([
        np.clip(np.random.normal(s1_mm,   s1_ms,   n_s1),  0, None),
        np.clip(np.random.normal(tm_mean, tm_std,  n_tgt), 0, None)
    ])

    dist_sq = ((cond - tc_mean) / max(tc_std, 0.01))**2 + \
              ((med  - tm_mean) / max(tm_std, 0.01))**2
    weights = np.exp(-0.5 * dist_sq)
    weights = weights / weights.sum()
    return cond, med, weights


# ─────────────────────────────────────────
# POWER LAW
# ─────────────────────────────────────────
def power_law_exponent(G):
    degrees = np.array([d for _, d in G.degree() if d > 0])
    counts, bins = np.histogram(degrees, bins=50)
    bin_centers  = (bins[:-1] + bins[1:]) / 2
    nonzero = counts > 0
    try:
        slope, _, r, _, _ = stats.linregress(
            np.log(bin_centers[nonzero]),
            np.log(counts[nonzero])
        )
        return slope, r**2
    except:
        return np.nan, np.nan


# ─────────────────────────────────────────
# MAIN PER-STRATUM RUN
# ─────────────────────────────────────────
def run_per_stratum(G, features, strata, treatments, ranges):
    print(f"\n{'='*65}")
    print("PER-STRATUM METRICS  (Absolute | Relative% | KL | Entropy)")
    print(f"{'='*65}")

    s1_nodes = strata['S1']
    base_pl, base_r2 = power_law_exponent(G)
    print(f"\n  Original power law: {base_pl:.4f}  R²={base_r2:.3f}")

    # Real feature values for reference
    all_real_c = np.array([features[n]['condition']  for n in features])
    all_real_m = np.array([features[n]['medication'] for n in features])
    print(f"\n  REAL DATA REFERENCE:")
    print(f"    condition  — mean={all_real_c.mean():.1f}  "
          f"std={all_real_c.std():.1f}  "
          f"range=[{all_real_c.min():.0f}, {all_real_c.max():.0f}]")
    print(f"    medication — mean={all_real_m.mean():.1f}  "
          f"std={all_real_m.std():.1f}  "
          f"range=[{all_real_m.min():.0f}, {all_real_m.max():.0f}]")

    results  = {}
    G_augmented = G.copy()

    for stratum_name in ['S1', 'S2', 'S3', 'S4', 'S5']:
        target_nodes = strata[stratum_name]
        if not target_nodes:
            continue

        treat, epsilon, desc = treatments[stratum_name]
        n_synth = len(target_nodes)

        true_c = np.array([features[n]['condition']  for n in target_nodes])
        true_m = np.array([features[n]['medication'] for n in target_nodes])
        true_cm = true_c.mean()
        true_mm = true_m.mean()

        # ── S1, S2: standard DP ──────────────────────────────────
        if treat == 'dp_only':
            _, abs_mae, rel_mae, noise_abs, noise_rel, kl_c, kl_m, h_c, h_m = \
                apply_standard_dp(target_nodes, features, epsilon, ranges)

            pl, r2 = power_law_exponent(G)
            pl_delta = abs(pl - base_pl)
            pl_delta_pct = 100 * pl_delta / abs(base_pl)

            results[stratum_name] = {
                'n_patients':    len(target_nodes),
                'epsilon':       epsilon,
                'treatment':     treat,
                'synth_added':   0,
                'power_law':     pl,
                'pl_delta':      pl_delta,
                'pl_delta_pct':  pl_delta_pct,
                'abs_mae':       abs_mae,
                'rel_mae':       rel_mae,
                'noise_abs':     noise_abs,
                'noise_rel':     noise_rel,
                'kl_c':          kl_c,
                'kl_m':          kl_m,
                'entropy_c':     h_c,
                'entropy_m':     h_m,
                'true_c_mean':   true_cm,
                'true_m_mean':   true_mm,
                'synth_c_mean':  None,
                'synth_m_mean':  None,
            }

            print(f"\n  {'─'*55}")
            print(f"  {stratum_name} | {desc}")
            print(f"  {'─'*55}")
            print(f"    Patients: {len(target_nodes):,}  |  Synthetic added: 0")
            print(f"    Real mean — condition: {true_cm:.1f}  medication: {true_mm:.1f}")
            print(f"    Power law: {pl:.4f}  (Δ={pl_delta:.4f} = {pl_delta_pct:.1f}%)")
            print(f"    Utility MAE:  abs={abs_mae:.3f}  rel={rel_mae:.3f}%")
            print(f"    Noise error:  abs={noise_abs:.3f}  rel={noise_rel:.3f}%")
            print(f"    KL divergence — condition: {kl_c:.4f}  medication: {kl_m:.4f}")
            print(f"    Shannon entropy — H(condition)={h_c:.3f}  H(medication)={h_m:.3f}")
            continue

        # ── S3, S4, S5: weighted crowd ───────────────────────────
        s1_frac = {'weighted_medium':  0.30,
                   'weighted_heavy':   0.15,
                   'weighted_maximum': 0.05}[treat]

        tc_mean, tc_std, tm_mean, tm_std = dp_stats(target_nodes, features, epsilon)

        cond, med, weights = weighted_crowd(
            s1_nodes, features,
            tc_mean, tc_std, tm_mean, tm_std,
            n_synth, epsilon, s1_frac
        )

        synth_ids = [f"synth_{stratum_name}_{i}" for i in range(n_synth)]
        for i, sid in enumerate(synth_ids):
            G_augmented.add_node(sid)
            neighbors = np.random.choice(target_nodes,
                                         min(3, len(target_nodes)),
                                         replace=False)
            for nb in neighbors:
                G_augmented.add_edge(sid, nb)

        pl, r2   = power_law_exponent(G_augmented)
        pl_delta = abs(pl - base_pl)
        pl_delta_pct = 100 * pl_delta / abs(base_pl)

        # Absolute MAE
        abs_mae_c = abs(np.mean(cond) - true_cm)
        abs_mae_m = abs(np.mean(med)  - true_mm)
        abs_mae   = (abs_mae_c + abs_mae_m) / 2

        # Relative MAE (%)
        rel_mae_c = 100 * abs_mae_c / max(ranges['condition'],  1)
        rel_mae_m = 100 * abs_mae_m / max(ranges['medication'], 1)
        rel_mae   = (rel_mae_c + rel_mae_m) / 2

        # Noise error
        noise_abs = abs(tc_mean - true_cm) + abs(tm_mean - true_mm)
        noise_rel = 100 * noise_abs / (ranges['condition'] + ranges['medication'])

        # KL divergence: synthetic vs real
        kl_c = kl_divergence(true_c, cond)
        kl_m = kl_divergence(true_m, med)

        # Entropy of real vs synthetic
        h_real_c  = entropy(true_c)
        h_real_m  = entropy(true_m)
        h_synth_c = entropy(cond)
        h_synth_m = entropy(med)

        results[stratum_name] = {
            'n_patients':    len(target_nodes),
            'epsilon':       epsilon,
            'treatment':     treat,
            'synth_added':   n_synth,
            'power_law':     pl,
            'pl_delta':      pl_delta,
            'pl_delta_pct':  pl_delta_pct,
            'abs_mae':       abs_mae,
            'rel_mae':       rel_mae,
            'noise_abs':     noise_abs,
            'noise_rel':     noise_rel,
            'kl_c':          kl_c,
            'kl_m':          kl_m,
            'entropy_c':     h_real_c,
            'entropy_m':     h_real_m,
            'entropy_synth_c': h_synth_c,
            'entropy_synth_m': h_synth_m,
            'true_c_mean':   true_cm,
            'true_m_mean':   true_mm,
            'synth_c_mean':  float(np.mean(cond)),
            'synth_m_mean':  float(np.mean(med)),
        }

        print(f"\n  {'─'*55}")
        print(f"  {stratum_name} | {desc}")
        print(f"  {'─'*55}")
        print(f"    Patients: {len(target_nodes):,}  |  Synthetic added: {n_synth:,}")
        print(f"    Real mean  — condition: {true_cm:.1f}  medication: {true_mm:.1f}")
        print(f"    Synth mean — condition: {np.mean(cond):.1f}  medication: {np.mean(med):.1f}")
        print(f"    Power law: {pl:.4f}  (Δ={pl_delta:.4f} = {pl_delta_pct:.1f}%)")
        print(f"    Utility MAE:  abs={abs_mae:.2f}  "
              f"rel_cond={rel_mae_c:.2f}%  rel_med={rel_mae_m:.2f}%")
        print(f"    Noise error:  abs={noise_abs:.3f}  rel={noise_rel:.4f}%")
        print(f"    KL divergence — condition: {kl_c:.4f}  medication: {kl_m:.4f}")
        print(f"    Entropy real  — H(cond)={h_real_c:.3f}  H(med)={h_real_m:.3f}")
        print(f"    Entropy synth — H(cond)={h_synth_c:.3f}  H(med)={h_synth_m:.3f}")

    return results, base_pl


# ─────────────────────────────────────────
# PLOT — 3 rows: absolute | relative | info theory
# ─────────────────────────────────────────
def plot_results(results, base_pl, ranges):
    strata_names = list(results.keys())
    colors = {
        'S1': '#4CAF50', 'S2': '#2196F3',
        'S3': '#FF9800', 'S4': '#F44336', 'S5': '#9C27B0'
    }
    col_list = [colors[s] for s in strata_names]

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    fig.suptitle(
        '5-Stratum Results — Absolute | Relative | Information Theory\n'
        'Synthea Patient Graph  |  S1–S2: Standard DP  |  S3–S5: Weighted Crowd Blending',
        fontsize=12, fontweight='bold'
    )

    def shade(ax):
        ax.axvspan(1.5, 4.5, alpha=0.07, color='red', label='Crowd zone')

    # ── ROW 1: Absolute metrics ──────────────────────────────
    # 1a: Power law absolute
    ax = axes[0, 0]
    ax.bar(strata_names, [results[s]['power_law'] for s in strata_names],
           color=col_list, alpha=0.85)
    ax.axhline(base_pl, color='black', linestyle='--', linewidth=2,
               label=f'Original ({base_pl:.3f})')
    shade(ax)
    ax.set_title('Power Law Exponent\n(absolute)')
    ax.set_ylabel('Exponent')
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    # 1b: Power law delta absolute
    ax = axes[0, 1]
    ax.bar(strata_names, [results[s]['pl_delta'] for s in strata_names],
           color=col_list, alpha=0.85)
    shade(ax)
    ax.set_title('Power Law Δ\n(absolute)')
    ax.set_ylabel('|Δ|')
    ax.grid(axis='y', alpha=0.3)

    # 1c: MAE absolute
    ax = axes[0, 2]
    ax.bar(strata_names, [results[s]['abs_mae'] for s in strata_names],
           color=col_list, alpha=0.85)
    shade(ax)
    ax.set_title('Utility MAE\n(absolute)')
    ax.set_ylabel('MAE')
    ax.grid(axis='y', alpha=0.3)

    # 1d: Noise error absolute
    ax = axes[0, 3]
    ax.bar(strata_names, [results[s]['noise_abs'] for s in strata_names],
           color=col_list, alpha=0.85)
    shade(ax)
    ax.set_title('Noise Error\n(absolute)')
    ax.set_ylabel('Error')
    ax.grid(axis='y', alpha=0.3)

    # ── ROW 2: Relative metrics (%) ──────────────────────────
    # 2a: Power law delta %
    ax = axes[1, 0]
    ax.bar(strata_names, [results[s]['pl_delta_pct'] for s in strata_names],
           color=col_list, alpha=0.85)
    shade(ax)
    ax.axhline(5, color='green', linestyle=':', linewidth=1.5, label='5% threshold')
    ax.axhline(10, color='orange', linestyle=':', linewidth=1.5, label='10% threshold')
    ax.set_title('Power Law Δ\n(relative %)')
    ax.set_ylabel('% change from original')
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    # 2b: MAE relative %
    ax = axes[1, 1]
    rel_mae_vals = [results[s]['rel_mae'] for s in strata_names]
    ax.bar(strata_names, rel_mae_vals, color=col_list, alpha=0.85)
    shade(ax)
    ax.axhline(5, color='green', linestyle=':', linewidth=1.5, label='5% threshold')
    ax.set_title('Utility MAE\n(relative % of feature range)')
    ax.set_ylabel('% of range')
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    # 2c: Noise error relative %
    ax = axes[1, 2]
    ax.bar(strata_names, [results[s]['noise_rel'] for s in strata_names],
           color=col_list, alpha=0.85)
    shade(ax)
    ax.set_title('Noise Error\n(relative % of feature range)')
    ax.set_ylabel('% of range')
    ax.grid(axis='y', alpha=0.3)

    # 2d: Real vs Synthetic mean (crowd blending strata only)
    ax = axes[1, 3]
    crowd_strata = [s for s in strata_names
                    if results[s]['synth_c_mean'] is not None]
    x = np.arange(len(crowd_strata))
    w = 0.35
    real_c  = [results[s]['true_c_mean']  for s in crowd_strata]
    synth_c = [results[s]['synth_c_mean'] for s in crowd_strata]
    ax.bar(x - w/2, real_c,  w, label='Real mean',  color='steelblue', alpha=0.85)
    ax.bar(x + w/2, synth_c, w, label='Synth mean', color='coral',     alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(crowd_strata)
    ax.set_title('Real vs Synthetic Mean\n(condition_count, crowd strata)')
    ax.set_ylabel('condition_count mean')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # ── ROW 3: Information Theory (Cover) ────────────────────
    # 3a: KL divergence condition
    ax = axes[2, 0]
    ax.bar(strata_names, [results[s]['kl_c'] for s in strata_names],
           color=col_list, alpha=0.85)
    shade(ax)
    ax.set_title('KL Divergence\nD_KL(real || noisy/synth) — condition')
    ax.set_ylabel('KL divergence (nats)')
    ax.grid(axis='y', alpha=0.3)

    # 3b: KL divergence medication
    ax = axes[2, 1]
    ax.bar(strata_names, [results[s]['kl_m'] for s in strata_names],
           color=col_list, alpha=0.85)
    shade(ax)
    ax.set_title('KL Divergence\nD_KL(real || noisy/synth) — medication')
    ax.set_ylabel('KL divergence (nats)')
    ax.grid(axis='y', alpha=0.3)

    # 3c: Shannon entropy real vs synthetic
    ax = axes[2, 2]
    crowd_strata_all = [s for s in strata_names
                        if 'entropy_synth_c' in results[s]]
    dp_strata        = [s for s in strata_names
                        if 'entropy_synth_c' not in results[s]]

    all_s  = strata_names
    h_real = [results[s]['entropy_c'] for s in all_s]
    h_syn  = [results[s].get('entropy_synth_c', results[s]['entropy_c'])
              for s in all_s]

    x2 = np.arange(len(all_s))
    ax.bar(x2 - w/2, h_real, w, label='Real H(X)',  color='steelblue', alpha=0.85)
    ax.bar(x2 + w/2, h_syn,  w, label='Synth H(X)', color='coral',     alpha=0.85)
    ax.set_xticks(x2)
    ax.set_xticklabels(all_s)
    ax.set_title('Shannon Entropy H(X)\nReal vs Noisy/Synthetic — condition')
    ax.set_ylabel('Entropy (bits)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # 3d: Privacy-Utility scatter (relative)
    ax = axes[2, 3]
    for s in strata_names:
        marker = 's' if results[s]['treatment'] != 'dp_only' else 'o'
        ax.scatter(results[s]['noise_rel'], results[s]['rel_mae'],
                   color=colors[s], s=200, zorder=5,
                   label=f"{s} (ε={results[s]['epsilon']})",
                   marker=marker)
        ax.annotate(s,
                    (results[s]['noise_rel'], results[s]['rel_mae']),
                    fontsize=9, fontweight='bold',
                    xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Noise Error (% of feature range)')
    ax.set_ylabel('Utility MAE (% of feature range)')
    ax.set_title('Privacy-Utility Tradeoff\n(relative %)  bottom-left = best\n'
                 '● DP only  ■ Crowd blending')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(SAVE_PATH + 'five_strata_info_theory.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n✅ five_strata_info_theory.png saved!")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    G, features, composite, nodes_df, ranges = load_synthea()

    strata, treatments, iqr_upper = define_five_strata(composite)

    results, base_pl = run_per_stratum(G, features, strata, treatments, ranges)

    plot_results(results, base_pl, ranges)

    print("\n" + "="*65)
    print("INTERPRETATION GUIDE:")
    print()
    print("  Power law Δ%  < 5%  → excellent preservation")
    print("  Power law Δ%  5-15% → acceptable")
    print("  Power law Δ%  > 15% → concerning")
    print()
    print("  KL divergence ≈ 0   → synthetic ≈ real distribution")
    print("  KL divergence > 1   → distributions clearly different")
    print()
    print("  Entropy match       → synthetic has same spread as real")
    print("  Entropy mismatch    → synthetic too uniform or too sparse")
    print("="*65)