"""
GitHub Social Network — Signal-Preserving (Weighed) Crowd Blending
Manual loader (bypasses PyG 404 error on graphmining.ai).

Raw files from: https://archive.ics.uci.edu/dataset/588/github+musae
Place in:       graphs/github/
                  ├── musae_git_edges.csv
                  ├── musae_git_features.json
                  └── musae_git_target.csv
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import csv

np.random.seed(42)

# ─────────────────────────────────────────
# LOAD GITHUB
# ─────────────────────────────────────────
def load_github(
    edges_path    = "graphs/github/musae_git_edges.csv",
    features_path = "graphs/github/musae_git_features.json",
    target_path   = "graphs/github/musae_git_target.csv",
):
    print("Loading features...")
    with open(features_path, 'r') as f:
        raw_features = json.load(f)

    node_ids  = [int(k) for k in raw_features.keys()]
    feat_raw  = np.array([len(raw_features[str(n)]) for n in node_ids], dtype=float)
    feat_norm = (feat_raw - feat_raw.min()) / (feat_raw.max() - feat_raw.min())
    features  = {node_ids[i]: feat_norm[i] for i in range(len(node_ids))}

    print("Loading labels...")
    labels = {}
    with open(target_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[int(row['id'])] = int(row['ml_target'])

    print("Building graph...")
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    with open(edges_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src, dst = int(row['id_1']), int(row['id_2'])
            if src in features and dst in features:
                G.add_edge(src, dst)

    print(f"\nGitHub loaded: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")
    print(f"  Feature range : [{feat_norm.min():.4f}, {feat_norm.max():.4f}]")
    print(f"  Classes       : 0=web dev ({sum(v==0 for v in labels.values()):,})  "
          f"1=ML dev ({sum(v==1 for v in labels.values()):,})")
    return G, features, labels


# ─────────────────────────────────────────
# STRATIFICATION — IQR
# ─────────────────────────────────────────
def stratify_by_iqr(G):
    degrees     = np.array([d for _, d in G.degree()])
    degree_dict = dict(G.degree())
    Q1  = np.percentile(degrees, 25)
    Q3  = np.percentile(degrees, 75)
    IQR = Q3 - Q1
    threshold = Q3 + 1.5 * IQR

    s1 = [n for n, d in degree_dict.items() if d < Q1]
    s2 = [n for n, d in degree_dict.items() if Q1 <= d <= threshold]
    s3 = [n for n, d in degree_dict.items() if d > threshold]

    print(f"\n  Q1={Q1:.1f}  Q3={Q3:.1f}  IQR={IQR:.1f}  threshold={threshold:.1f}")
    print(f"  S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}")
    return s1, s2, s3, threshold


# ─────────────────────────────────────────
# DP RELEASE OF S3 STATS
# ─────────────────────────────────────────
def dp_release_s3_stats(s3_nodes, features, epsilon_s3_stats=0.1):
    vals        = np.array([features[n] for n in s3_nodes])
    sensitivity = 1.0 / len(s3_nodes)

    noisy_mean = float(np.clip(
        np.mean(vals) + np.random.laplace(0, sensitivity / epsilon_s3_stats),
        0, 1
    ))
    noisy_std = float(max(0.01,
        np.std(vals) + np.random.laplace(0, sensitivity / epsilon_s3_stats)
    ))

    print(f"\n  S3 true mean:  {np.mean(vals):.4f}  →  DP mean: {noisy_mean:.4f}")
    print(f"  S3 true std:   {np.std(vals):.4f}  →  DP std:  {noisy_std:.4f}")
    print(f"  ε spent on S3 stats: {epsilon_s3_stats}")
    return noisy_mean, noisy_std


# ─────────────────────────────────────────
# APPROACH A — FLAT CROWD
# ─────────────────────────────────────────
def generate_flat_crowd(s1_nodes, features, n_synthetic, epsilon=0.2):
    vals        = np.array([features[n] for n in s1_nodes])
    sensitivity = 1.0 / len(s1_nodes)

    noisy_mean = float(np.clip(
        np.mean(vals) + np.random.laplace(0, sensitivity / epsilon), 0, 1
    ))
    noisy_std = float(max(0.01,
        np.std(vals) + np.random.laplace(0, sensitivity / epsilon)
    ))

    scores  = np.clip(np.random.normal(noisy_mean, noisy_std, n_synthetic), 0, 1)
    weights = np.ones(n_synthetic)

    synth = {f"flat_{i}": {"score": scores[i], "weight": weights[i]}
             for i in range(n_synthetic)}

    print(f"\n  [FLAT]  mean={noisy_mean:.4f}  std={noisy_std:.4f}  n={n_synthetic}")
    return synth, noisy_mean, noisy_std


# ─────────────────────────────────────────
# APPROACH B — WEIGHTED CROWD
# ─────────────────────────────────────────
def generate_weighted_crowd(s1_nodes, features,
                             s3_noisy_mean, s3_noisy_std,
                             n_synthetic, epsilon_s1=0.2,
                             s1_fraction=0.3):
    s1_vals     = np.array([features[n] for n in s1_nodes])
    sensitivity = 1.0 / len(s1_nodes)
    s1_noisy_mean = float(np.clip(
        np.mean(s1_vals) + np.random.laplace(0, sensitivity / epsilon_s1), 0, 1
    ))
    s1_noisy_std = float(max(0.01,
        np.std(s1_vals) + np.random.laplace(0, sensitivity / epsilon_s1)
    ))

    n_from_s1 = int(n_synthetic * s1_fraction)
    n_from_s3 = n_synthetic - n_from_s1

    scores_s1 = np.clip(
        np.random.normal(s1_noisy_mean, s1_noisy_std, n_from_s1), 0, 1
    )
    scores_s3 = np.clip(
        np.random.normal(s3_noisy_mean, s3_noisy_std, n_from_s3), 0, 1
    )
    all_scores = np.concatenate([scores_s1, scores_s3])

    weights = np.exp(
        -((all_scores - s3_noisy_mean) ** 2) / (2 * s3_noisy_std ** 2)
    )
    weights = weights / weights.sum()

    # Weighted mean = the true signal of where the crowd lands
    weighted_mean = float(np.sum(all_scores * weights))
    weighted_std  = float(np.sqrt(np.sum(weights * (all_scores - weighted_mean) ** 2)))

    near_s3_mask   = np.abs(all_scores - s3_noisy_mean) <= s3_noisy_std
    near_s3_mean_w = weights[near_s3_mask].mean() if near_s3_mask.any() else 0.0

    print(f"\n  [WEIGHTED]  S1-based: {n_from_s1:,}  S3-targeted: {n_from_s3:,}")
    print(f"  S1 mean={s1_noisy_mean:.4f}  S3 mean={s3_noisy_mean:.4f}  "
          f"← {'S3 < S1: hub nodes sparse' if s3_noisy_mean < s1_noisy_mean else 'S3 > S1: hub nodes dense'}")
    print(f"  Raw score range:    [{all_scores.min():.3f}, {all_scores.max():.3f}]")
    print(f"  Weighted mean:      {weighted_mean:.4f}  (target S3 mean: {s3_noisy_mean:.4f})")
    print(f"  Weighted std:       {weighted_std:.4f}")
    print(f"  Near-S3 band: [{s3_noisy_mean - s3_noisy_std:.3f}, "
          f"{s3_noisy_mean + s3_noisy_std:.3f}]  "
          f"({near_s3_mask.sum()} nodes)  mean_weight={near_s3_mean_w:.6f}")

    synth = {f"weighted_{i}": {"score": all_scores[i], "weight": weights[i]}
             for i in range(n_synthetic)}
    return synth, weighted_mean, weighted_std


# ─────────────────────────────────────────
# CLASS BALANCE PER STRATUM
# ─────────────────────────────────────────
def class_balance_per_stratum(labels, s1, s2, s3):
    print("\n─── Class balance per stratum (0=web dev, 1=ML dev) ───")
    for name, idx in [("S1", s1), ("S2", s2), ("S3", s3)]:
        if not idx:
            continue
        stratum_labels = [labels[n] for n in idx]
        n0 = stratum_labels.count(0)
        n1 = stratum_labels.count(1)
        print(f"  {name}: web_dev={n0:,} ({100*n0/len(idx):.1f}%)  "
              f"ml_dev={n1:,} ({100*n1/len(idx):.1f}%)")


# ─────────────────────────────────────────
# COMPARE FLAT vs WEIGHTED
# Primary metric: dist_to_S3_mean (dataset-agnostic)
# ─────────────────────────────────────────
def compare_approaches(G, features, labels):
    print("\n" + "=" * 65)
    print("FLAT vs WEIGHTED CROWD BLENDING — GITHUB DATASET")
    print("=" * 65)

    s1, s2, s3, threshold = stratify_by_iqr(G)
    class_balance_per_stratum(labels, s1, s2, s3)

    print("\n─── DP release S3 stats ───")
    s3_mean_dp, s3_std_dp = dp_release_s3_stats(s3, features, epsilon_s3_stats=0.1)

    s1_mean   = np.mean([features[n] for n in s1])
    s3_is_low = s3_mean_dp < s1_mean

    print(f"\n  Direction: S3 is {'LOW (hub nodes sparse)' if s3_is_low else 'HIGH (hub nodes dense)'}")
    print(f"  S1 mean={s1_mean:.4f}  S3 mean={s3_mean_dp:.4f}  "
          f"gap={abs(s3_mean_dp - s1_mean):.4f}")

    N_SYNTHETIC = 500

    print("\n─── Approach A: Flat Crowd ───")
    flat_crowd, flat_mean, flat_std = generate_flat_crowd(
        s1, features, N_SYNTHETIC, epsilon=0.2
    )

    print("\n─── Approach B: Weighted Crowd ───")
    weighted_crowd, weighted_mean, weighted_std = generate_weighted_crowd(
        s1, features, s3_mean_dp, s3_std_dp,
        N_SYNTHETIC, epsilon_s1=0.2, s1_fraction=0.3
    )

    # ── PRIMARY METRIC: distance of crowd mean to S3 mean ──────────────────
    flat_dist     = abs(flat_mean     - s3_mean_dp)
    weighted_dist = abs(weighted_mean - s3_mean_dp)
    improvement   = flat_dist - weighted_dist   # positive = weighted is better

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  S3 mean (DP)             : {s3_mean_dp:.4f}  "
          f"({'LOW — hub nodes sparse' if s3_is_low else 'HIGH — hub nodes dense'})")
    print(f"  S1 mean                  : {s1_mean:.4f}")
    print(f"  Gap S1→S3                : {abs(s3_mean_dp - s1_mean):.4f}")
    print()
    print(f"  Flat crowd mean          : {flat_mean:.4f}  "
          f"(dist to S3: {flat_dist:.4f})")
    print(f"  Weighted crowd mean      : {weighted_mean:.4f}  "
          f"(dist to S3: {weighted_dist:.4f})")
    print()
    print(f"  Improvement (dist saved) : {improvement:+.4f}  "
          f"{'← weighted wins ✓' if improvement > 0 else '← flat wins'}")
    print(f"  % gap closed by weighted : "
          f"{100 * improvement / abs(s3_mean_dp - s1_mean):.1f}%")

    plot_comparison(features, s1, s2, s3,
                    flat_crowd, weighted_crowd,
                    flat_mean, weighted_mean,
                    threshold, s3_mean_dp, s3_std_dp, s1_mean)

    return flat_mean, weighted_mean, s3_mean_dp


# ─────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────
def plot_comparison(features, s1, s2, s3,
                    flat_crowd, weighted_crowd,
                    flat_mean, weighted_mean,
                    threshold, s3_mean_dp, s3_std_dp, s1_mean):

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("GitHub — Flat vs Weighted Crowd Blending", fontsize=13)

    s1_vals = [features[n] for n in s1]
    s2_vals = [features[n] for n in s2]
    s3_vals = [features[n] for n in s3]

    flat_scores     = np.array([v["score"]  for v in flat_crowd.values()])
    weighted_scores = np.array([v["score"]  for v in weighted_crowd.values()])
    weighted_w      = np.array([v["weight"] for v in weighted_crowd.values()])

    # Panel 1 — Real node features
    ax = axes[0]
    ax.hist(s1_vals, bins=40, alpha=0.6, color='steelblue',
            label=f'S1 ({len(s1):,})')
    ax.hist(s2_vals, bins=40, alpha=0.5, color='mediumseagreen',
            label=f'S2 ({len(s2):,})')
    ax.hist(s3_vals, bins=40, alpha=0.7, color='tomato',
            label=f'S3 ({len(s3):,})')
    ax.axvline(threshold,  color='black', linestyle='--', linewidth=1.2,
               label='IQR threshold')
    ax.axvline(s3_mean_dp, color='red',   linestyle=':', linewidth=1.5,
               label=f'S3 mean={s3_mean_dp:.3f}')
    ax.axvline(s1_mean,    color='steelblue', linestyle=':', linewidth=1.5,
               label=f'S1 mean={s1_mean:.3f}')
    ax.set_title('Real Node Features by Stratum')
    ax.set_xlabel('Normalised Feature Score')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)

    # Panel 2 — Synthetic distributions with crowd MEANS as primary marker
    ax2 = axes[1]
    ax2.hist(flat_scores,     bins=40, alpha=0.6, color='steelblue', label='Flat')
    ax2.hist(weighted_scores, bins=40, alpha=0.6, color='tomato',    label='Weighted')
    # Crowd means — the primary metric
    ax2.axvline(flat_mean,     color='blue', linestyle='--', linewidth=2.0,
                label=f'Flat mean={flat_mean:.3f}')
    ax2.axvline(weighted_mean, color='red',  linestyle='--', linewidth=2.0,
                label=f'Weighted mean={weighted_mean:.3f}')
    # S3 target
    ax2.axvline(s3_mean_dp, color='black', linestyle=':', linewidth=1.5,
                label=f'S3 mean={s3_mean_dp:.3f}  ← target')
    ax2.set_title('Synthetic Score Distributions\n(dashed = crowd mean, dotted = S3 target)')
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Count')
    ax2.legend(fontsize=8)

    # Panel 3 — Score vs Weight
    ax3 = axes[2]
    sc = ax3.scatter(weighted_scores, weighted_w,
                     c=weighted_scores, cmap='RdYlGn', alpha=0.4, s=10)
    band_lo = max(0.0, s3_mean_dp - s3_std_dp)
    band_hi = min(1.0, s3_mean_dp + s3_std_dp)
    ax3.axvspan(band_lo, band_hi, alpha=0.15, color='tomato',
                label=f'Near-S3 band\n[{band_lo:.2f}, {band_hi:.2f}]')
    ax3.axvline(s3_mean_dp, color='red', linestyle='--',
                linewidth=1.2, label=f'S3 mean={s3_mean_dp:.3f}')
    ax3.set_title('Weighted Crowd: Score vs Weight')
    ax3.set_xlabel('Synthetic Node Score')
    ax3.set_ylabel('Weight (Gaussian kernel)')
    ax3.legend(fontsize=7)
    plt.colorbar(sc, ax=ax3, label='Score')

    plt.tight_layout()
    plt.savefig('github_crowd_blending.png', dpi=150)
    plt.show()
    print("\nPlot saved → github_crowd_blending.png")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    G, features, labels = load_github()
    flat_mean, weighted_mean, s3_mean = compare_approaches(G, features, labels)
