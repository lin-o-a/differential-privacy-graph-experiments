import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

np.random.seed(42)

# ─────────────────────────────────────────
# LOAD CORA
# ─────────────────────────────────────────
def load_cora(cora_content_path="cora/cora.content",
              cora_cites_path="cora/cora.cites"):
    node_ids, feature_matrix, labels = [], [], {}
    with open(cora_content_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            node_id = int(parts[0])
            binary_features = np.array([int(x) for x in parts[1:-1]])
            node_ids.append(node_id)
            feature_matrix.append(np.sum(binary_features))
            labels[node_id] = parts[-1]

    feat_array = np.array(feature_matrix, dtype=float)
    feat_norm = (feat_array - feat_array.min()) / (feat_array.max() - feat_array.min())
    features = {node_ids[i]: feat_norm[i] for i in range(len(node_ids))}

    G = nx.Graph()
    G.add_nodes_from(node_ids)
    with open(cora_cites_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            src, dst = int(parts[0]), int(parts[1])
            if src in features and dst in features:
                G.add_edge(src, dst)

    print(f"Cora loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, features, labels


# ─────────────────────────────────────────
# STRATIFICATION — IQR
# ─────────────────────────────────────────
def stratify_by_iqr(G):
    degrees    = np.array([d for _, d in G.degree()])
    degree_dict = dict(G.degree())
    Q1  = np.percentile(degrees, 25)
    Q3  = np.percentile(degrees, 75)
    IQR = Q3 - Q1
    threshold = Q3 + 1.5 * IQR

    s1 = [n for n, d in degree_dict.items() if d < Q1]
    s2 = [n for n, d in degree_dict.items() if Q1 <= d <= threshold]
    s3 = [n for n, d in degree_dict.items() if d > threshold]

    print(f"\n  Q1={Q1:.1f}  Q3={Q3:.1f}  IQR={IQR:.1f}  threshold={threshold:.1f}")
    print(f"  S1={len(s1)}  S2={len(s2)}  S3={len(s3)}")
    return s1, s2, s3, threshold


# ─────────────────────────────────────────
# DP RELEASE OF S3 STATS (needed for weighted blending)
# ─────────────────────────────────────────
def dp_release_s3_stats(s3_nodes, features, epsilon_s3_stats=0.1):
    """
    Release DP-noised mean and std of S3 feature values.
    These are used to TARGET synthetic node sampling near S3.
    Costs a small extra epsilon on S3 budget.
    """
    vals = np.array([features[n] for n in s3_nodes])
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
# APPROACH A — FLAT CROWD (baseline)
# Each synthetic node sampled from S1 stats only.
# All weights = 1 (uniform).
# ─────────────────────────────────────────
def generate_flat_crowd(s1_nodes, features, n_synthetic, epsilon=0.2):
    """
    Baseline: synthetic nodes drawn from S1 distribution.
    No targeting toward S3. All weights equal.
    """
    vals = np.array([features[n] for n in s1_nodes])
    sensitivity = 1.0 / len(s1_nodes)

    noisy_mean = float(np.clip(
        np.mean(vals) + np.random.laplace(0, sensitivity / epsilon), 0, 1
    ))
    noisy_std = float(max(0.01,
        np.std(vals) + np.random.laplace(0, sensitivity / epsilon)
    ))

    scores  = np.clip(np.random.normal(noisy_mean, noisy_std, n_synthetic), 0, 1)
    weights = np.ones(n_synthetic)          # ← uniform, no targeting

    synth = {f"flat_{i}": {"score": scores[i], "weight": weights[i]}
             for i in range(n_synthetic)}

    print(f"\n  [FLAT]  mean={noisy_mean:.4f}  std={noisy_std:.4f}  n={n_synthetic}")
    return synth


# ─────────────────────────────────────────
# APPROACH B — WEIGHTED CROWD
#
# Two things happen here (answer to your question):
#
# 1. SCORES: 70% of synthetic nodes are SAMPLED near S3
#    (drawn from N(μ_S3_dp, σ_S3_dp) instead of S1)
#    → their feature scores ARE concentrated near outlier range
#
# 2. WEIGHTS: every synthetic node gets a weight
#    w_i = exp(-(v_i - μ_S3)² / (2σ_S3²))
#    → nodes closer to S3 boundary count MORE in the pool
#    → nodes far from S3 boundary count LESS
#
# These two mechanisms work together:
#   Sampling  → puts synthetic nodes IN the right place
#   Weighting → makes nodes IN the right place count MORE
# ─────────────────────────────────────────
def generate_weighted_crowd(s1_nodes, features,
                             s3_noisy_mean, s3_noisy_std,
                             n_synthetic, epsilon_s1=0.2,
                             s1_fraction=0.3):
    """
    Weighted crowd blending:

    s1_fraction = fraction of synthetic nodes drawn from S1
                  (1 - s1_fraction) drawn from near-S3 distribution

    Weight formula per synthetic node i:
        w_i = exp( -(v_i - μ_S3)² / (2 * σ_S3²) )

    This is a Gaussian kernel centred on S3 mean.
    Nodes near S3 → w close to 1.0
    Nodes far from S3 → w close to 0.0
    """
    # ── S1 stats (DP released) ──────────────────────────────
    s1_vals = np.array([features[n] for n in s1_nodes])
    sensitivity = 1.0 / len(s1_nodes)
    s1_noisy_mean = float(np.clip(
        np.mean(s1_vals) + np.random.laplace(0, sensitivity / epsilon_s1), 0, 1
    ))
    s1_noisy_std = float(max(0.01,
        np.std(s1_vals) + np.random.laplace(0, sensitivity / epsilon_s1)
    ))

    # ── Sample scores ────────────────────────────────────────
    n_from_s1 = int(n_synthetic * s1_fraction)
    n_from_s3 = n_synthetic - n_from_s1

    # S1-based synthetic nodes (cover the normal range)
    scores_s1 = np.clip(
        np.random.normal(s1_noisy_mean, s1_noisy_std, n_from_s1), 0, 1
    )

    # S3-targeted synthetic nodes (scores ARE near outlier range)
    scores_s3 = np.clip(
        np.random.normal(s3_noisy_mean, s3_noisy_std, n_from_s3), 0, 1
    )

    all_scores = np.concatenate([scores_s1, scores_s3])

    # ── Calculate weights ────────────────────────────────────
    # w_i = exp( -(v_i - μ_S3)² / (2σ_S3²) )
    # Gaussian kernel: high weight near S3 mean, low weight far away
    weights = np.exp(
        -((all_scores - s3_noisy_mean) ** 2) / (2 * s3_noisy_std ** 2)
    )
    # Normalise so weights sum to 1
    weights = weights / weights.sum()

    synth = {f"weighted_{i}": {"score": all_scores[i], "weight": weights[i]}
             for i in range(n_synthetic)}

    print(f"\n  [WEIGHTED]  S1-based nodes: {n_from_s1}  S3-targeted nodes: {n_from_s3}")
    print(f"  Score range: [{all_scores.min():.3f}, {all_scores.max():.3f}]")
    print(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"  Mean weight near S3 (score > 0.7): "
          f"{weights[all_scores > 0.7].mean():.4f}")
    return synth


# ─────────────────────────────────────────
# CEILING RELEASE (works for both flat and weighted)
# ─────────────────────────────────────────
def release_ceiling(s3_nodes, synth_crowd, features, epsilon=0.5):
    s3_vals   = np.array([features[n] for n in s3_nodes])
    synth_vals = np.array([v["score"] for v in synth_crowd.values()])

    # Extend synthetic range slightly above S3 max
    s3_max = np.max(s3_vals)
    extended = np.append(
        synth_vals,
        np.random.uniform(s3_max * 0.95, min(s3_max * 1.05, 1.0), size=10)
    )
    mixed_pool = np.concatenate([s3_vals, np.clip(extended, 0, 1)])

    true_ceiling = np.percentile(mixed_pool, 95)
    noise        = np.random.laplace(0, (1.0 / len(s3_nodes)) / epsilon)
    noisy_ceiling = float(np.clip(true_ceiling + noise, 0, 1))

    return noisy_ceiling, abs(noise)


# ─────────────────────────────────────────
# COMPARE FLAT vs WEIGHTED
# ─────────────────────────────────────────
def compare_approaches(G, features):
    print("\n" + "=" * 65)
    print("FLAT vs WEIGHTED CROWD BLENDING — COMPARISON")
    print("=" * 65)

    s1_nodes, s2_nodes, s3_nodes, threshold = stratify_by_iqr(G)

    # ── DP release S3 stats (small extra budget for weighted) ──
    print("\n─── DP release S3 stats (for weighted crowd) ───")
    s3_mean_dp, s3_std_dp = dp_release_s3_stats(
        s3_nodes, features, epsilon_s3_stats=0.1
    )

    # ── Budget summary ─────────────────────────────────────────
    print(f"""
Budget allocation:
  Flat crowd:
    S1 stats release:   ε = 0.20
    S3 ceiling:         ε = 0.50
    Total (parallel):   ε = 0.50  (max of above)

  Weighted crowd:
    S1 stats release:   ε = 0.20
    S3 stats release:   ε = 0.10  ← extra cost for targeting
    S3 ceiling:         ε = 0.50
    Total (parallel):   ε = 0.50  (max of above, same budget!)
    """)

    # ── Generate both crowds ───────────────────────────────────
    print("─── Generating flat crowd ───")
    flat_crowd = generate_flat_crowd(
        s1_nodes, features, n_synthetic=80, epsilon=0.2
    )

    print("\n─── Generating weighted crowd ───")
    weighted_crowd = generate_weighted_crowd(
        s1_nodes, features,
        s3_noisy_mean=s3_mean_dp,
        s3_noisy_std=s3_std_dp,
        n_synthetic=80,
        epsilon_s1=0.2,
        s1_fraction=0.3
    )

    # ── Release ceilings ───────────────────────────────────────
    flat_ceiling, flat_noise       = release_ceiling(s3_nodes, flat_crowd,     features)
    weighted_ceiling, weighted_noise = release_ceiling(s3_nodes, weighted_crowd, features)

    print(f"\n  Flat ceiling:     {flat_ceiling:.4f}  (noise={flat_noise:.4f})")
    print(f"  Weighted ceiling: {weighted_ceiling:.4f}  (noise={weighted_noise:.4f})")

    # ── Plot comparison ────────────────────────────────────────
    plot_comparison(features, s1_nodes, s3_nodes,
                    flat_crowd, weighted_crowd,
                    flat_ceiling, weighted_ceiling,
                    s3_mean_dp)


# ─────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────
def plot_comparison(features, s1_nodes, s3_nodes,
                    flat_crowd, weighted_crowd,
                    flat_ceiling, weighted_ceiling,
                    s3_mean_dp):

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    s3_vals  = np.array([features[n] for n in s3_nodes])
    s1_vals  = np.array([features[n] for n in s1_nodes])

    flat_scores     = np.array([v["score"]  for v in flat_crowd.values()])
    weighted_scores = np.array([v["score"]  for v in weighted_crowd.values()])
    weighted_wts    = np.array([v["weight"] for v in weighted_crowd.values()])

    # ── Plot 1: Score distributions ───────────────────────────
    axes[0].hist(s1_vals,         bins=30, alpha=0.4, color='green',
                 label=f'S1 real (n={len(s1_nodes)})')
    axes[0].hist(s3_vals,         bins=20, alpha=0.6, color='red',
                 label=f'S3 real outliers (n={len(s3_nodes)})')
    axes[0].hist(flat_scores,     bins=30, alpha=0.5, color='steelblue',
                 label='Flat synthetic', linestyle='--')
    axes[0].hist(weighted_scores, bins=30, alpha=0.5, color='orange',
                 label='Weighted synthetic')
    axes[0].axvline(flat_ceiling,     color='steelblue', linestyle='--',
                    label=f'Flat ceiling={flat_ceiling:.3f}')
    axes[0].axvline(weighted_ceiling, color='darkorange', linestyle='-.',
                    label=f'Weighted ceiling={weighted_ceiling:.3f}')
    axes[0].set_title('Score Distributions\nFlat vs Weighted Synthetic Crowd')
    axes[0].set_xlabel('Normalised feature value')
    axes[0].legend(fontsize=7)

    # ── Plot 2: Weight distribution of weighted crowd ─────────
    axes[1].scatter(weighted_scores, weighted_wts,
                    c=weighted_scores, cmap='RdYlGn_r',
                    alpha=0.7, s=40)
    axes[1].axvline(s3_mean_dp, color='red', linestyle='--',
                    label=f'S3 DP mean={s3_mean_dp:.3f}')
    axes[1].set_title('Weighted Crowd:\nScore vs Weight per Synthetic Node')
    axes[1].set_xlabel('Synthetic node score')
    axes[1].set_ylabel('Weight  w_i')
    axes[1].legend(fontsize=8)
    axes[1].text(0.05, 0.92,
                 r'$w_i = \exp\left(-\frac{(v_i - \mu_{S3})^2}{2\sigma_{S3}^2}\right)$',
                 transform=axes[1].transAxes, fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # ── Plot 3: Density near S3 boundary ──────────────────────
    boundary_range = np.linspace(0, 1, 100)

    # KDE-style density count in sliding window
    def sliding_density(scores, weights=None, window=0.05):
        density = []
        for x in boundary_range:
            mask = (scores >= x - window) & (scores <= x + window)
            if weights is not None:
                density.append(weights[mask].sum())
            else:
                density.append(mask.sum() / len(scores))
        return np.array(density)

    flat_density     = sliding_density(flat_scores)
    weighted_density = sliding_density(weighted_scores, weighted_wts)

    # Normalise for comparison
    flat_density     = flat_density / flat_density.max()
    weighted_density = weighted_density / weighted_density.max()

    axes[2].plot(boundary_range, flat_density,
                 color='steelblue', linewidth=2, label='Flat crowd density')
    axes[2].plot(boundary_range, weighted_density,
                 color='darkorange', linewidth=2, label='Weighted crowd density')
    axes[2].axvspan(s3_vals.min(), s3_vals.max(),
                    alpha=0.15, color='red', label='S3 real range')
    axes[2].axvline(s3_mean_dp, color='red', linestyle='--', alpha=0.7)
    axes[2].set_title('Crowd Density Near S3 Boundary\n(higher = better cover)')
    axes[2].set_xlabel('Feature value')
    axes[2].set_ylabel('Normalised crowd density')
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('weighted_crowd_blending.png', dpi=150)
    plt.show()
    print("\n  Figure saved: weighted_crowd_blending.png")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("Loading Cora...")
    G, features, labels = load_cora()
    compare_approaches(G, features)
