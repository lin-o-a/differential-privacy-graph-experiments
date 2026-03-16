import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

np.random.seed(42)

# ─────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────

def min_max_normalize(features):
    """
    Normalize each feature dimension to [0, 1] independently.
    Formula: (x - x_min) / (x_max - x_min)
    """
    features = np.array(features, dtype=float)
    f_min = features.min(axis=0)   # min per column (per dimension)
    f_max = features.max(axis=0)   # max per column
    denom = f_max - f_min
    denom[denom == 0] = 1          # avoid division by zero if all values equal
    return (features - f_min) / denom


def add_gaussian_noise(features, sensitivity, epsilon):
    """
    Add Gaussian noise calibrated to global max sensitivity.
    sigma = sensitivity / epsilon
    Noise shape matches feature shape.
    """
    sigma = sensitivity / epsilon
    noise = np.random.normal(0, sigma, features.shape)
    return features + noise, noise, sigma


def compute_snr(features, sigma):
    """
    SNR per node = L2 norm of feature vector / sigma
    SNR > 1 means signal stronger than noise
    SNR < 1 means noise stronger than signal (node is destroyed)
    """
    norms = np.linalg.norm(features, axis=1)   # L2 norm per row (per node)
    return norms / sigma


def compute_error_percent(original, noisy):
    """
    Percentage error per node:
    || noisy - original ||_2 / || original ||_2 * 100
    Answers: noise moved features by X% of original magnitude
    """
    noise_magnitude = np.linalg.norm(noisy - original, axis=1)
    original_magnitude = np.linalg.norm(original, axis=1)
    # avoid division by zero
    safe_denom = np.where(original_magnitude == 0, 1e-9, original_magnitude)
    return (noise_magnitude / safe_denom) * 100


# ─────────────────────────────────────────────
# STEP 1 — BUILD TOY GRAPH (10 nodes, manual)
# ─────────────────────────────────────────────

def step1_build_toy_graph():
    print("\n" + "="*60)
    print("STEP 1 — TOY GRAPH (10 nodes)")
    print("="*60)

    G = nx.Graph()
    G.add_nodes_from(range(10))

    # Outlier nodes (0,1,2) get many edges → high degree
    edges = [
        (0,1),(0,2),(0,3),(0,4),(0,5),(0,6),   # node 0: degree 6
        (1,2),(1,3),(1,4),(1,5),                 # node 1: degree 5
        (2,3),(2,4),(2,5),                       # node 2: degree 4
        (3,4),(3,5),(3,6),(3,7),                 # node 3: degree 4+
        (4,5),(5,6),(6,7),(7,8),(8,9)            # normal connections
    ]
    G.add_edges_from(edges)

    # Features: outliers have large values, normals have small values
    # 3 dimensions: [activity_score, engagement, reach]
    raw_features = np.array([
        [8.5, 7.2, 9.1],   # node 0 - outlier
        [7.8, 8.0, 7.5],   # node 1 - outlier
        [6.9, 7.5, 8.2],   # node 2 - outlier
        [2.1, 1.8, 2.5],   # node 3 - normal
        [1.5, 2.2, 1.9],   # node 4 - normal
        [2.8, 1.5, 2.1],   # node 5 - normal
        [1.9, 2.7, 1.4],   # node 6 - normal
        [2.3, 1.6, 2.8],   # node 7 - normal
        [1.7, 2.4, 2.0],   # node 8 - normal
        [2.5, 1.9, 1.7],   # node 9 - normal
    ])

    # Normalize features before computing norms
    features = min_max_normalize(raw_features)

    node_types = ['OUTLIER']*3 + ['normal']*7

    print(f"\n  {'Node':>5} {'Degree':>7} {'Type':>9}  "
          f"{'Norm. features':>30}  {'L2 norm':>8}")
    print("  " + "-"*65)

    for i in G.nodes():
        deg  = G.degree(i)
        kind = node_types[i]
        norm = np.linalg.norm(features[i])
        print(f"  {i:>5} {deg:>7} {kind:>9}  "
              f"{str(np.round(features[i],2)):>30}  {norm:>8.4f}")

    return G, features, node_types


# ─────────────────────────────────────────────
# STEP 2 — BUILD REAL-SCALE GRAPH (500 nodes)
# ─────────────────────────────────────────────

def step2_build_real_graph():
    print("\n" + "="*60)
    print("STEP 2 — REAL-SCALE GRAPH (500 nodes, power law)")
    print("="*60)

    # Barabasi-Albert model generates power-law degree distribution
    # m=3 means each new node connects to 3 existing nodes
    G = nx.barabasi_albert_graph(500, m=3, seed=42)

    degrees = np.array([G.degree(i) for i in G.nodes()])
    mean_deg = degrees.mean()
    std_deg  = degrees.std()

    # Outlier threshold: degree > mean + 2*std
    outlier_threshold = mean_deg + 2 * std_deg
    node_types = [
        'OUTLIER' if degrees[i] > outlier_threshold else 'normal'
        for i in G.nodes()
    ]

    n_outliers = sum(1 for t in node_types if t == 'OUTLIER')
    print(f"\n  Nodes         : {G.number_of_nodes()}")
    print(f"  Edges         : {G.number_of_edges()}")
    print(f"  Mean degree   : {mean_deg:.2f}")
    print(f"  Std degree    : {std_deg:.2f}")
    print(f"  Max degree    : {degrees.max()}")
    print(f"  Outlier thresh: {outlier_threshold:.2f}  (mean + 2*std)")
    print(f"  Outliers      : {n_outliers} ({100*n_outliers/500:.1f}%)")
    print(f"  Normal nodes  : {500 - n_outliers} ({100*(500-n_outliers)/500:.1f}%)")

    # Generate features correlated with degree (realistic assumption)
    # Higher degree nodes → larger feature values
    raw_features = np.zeros((500, 3))
    for i in G.nodes():
        base = degrees[i] * 0.5          # feature magnitude scales with degree
        raw_features[i] = np.random.normal(base, 0.5, 3)
        raw_features[i] = np.abs(raw_features[i])   # keep positive

    features = min_max_normalize(raw_features)

    return G, features, node_types, degrees


# ─────────────────────────────────────────────
# STEP 3 — APPLY NOISE AND MEASURE DAMAGE
# ─────────────────────────────────────────────

def step3_apply_noise_and_measure(G, features, node_types, graph_label, epsilon=1.0):
    print(f"\n" + "="*60)
    print(f"STEP 3 — NOISE + DAMAGE MEASUREMENT: {graph_label}")
    print("="*60)

    degrees = np.array([G.degree(i) for i in G.nodes()])

    # Sensitivity = max degree (global worst case)
    sensitivity = degrees.max()
    print(f"\n  Epsilon       : {epsilon}")
    print(f"  Max degree    : {sensitivity}  ← this sets sigma for ALL nodes")

    noisy_features, noise, sigma = add_gaussian_noise(features, sensitivity, epsilon)

    print(f"  Sigma (noise) : {sigma:.4f}")

    # ── Per-node measurements ──
    snr           = compute_snr(features, sigma)
    error_percent = compute_error_percent(features, noisy_features)

    # ── Split by type ──
    outlier_mask = np.array([t == 'OUTLIER' for t in node_types])
    normal_mask  = ~outlier_mask

    def summarize(mask, label):
        snr_group   = snr[mask]
        err_group   = error_percent[mask]
        destroyed   = (snr_group < 1).sum()
        total       = mask.sum()
        print(f"\n  [{label}]  n={total}")
        print(f"    SNR   — mean: {snr_group.mean():.3f}  "
              f"min: {snr_group.min():.3f}  max: {snr_group.max():.3f}")
        print(f"    Error — mean: {err_group.mean():.1f}%  "
              f"min: {err_group.min():.1f}%  max: {err_group.max():.1f}%")
        print(f"    Nodes with SNR < 1 (destroyed): "
              f"{destroyed} / {total}  ({100*destroyed/total:.1f}%)")
        return snr_group, err_group

    snr_out, err_out = summarize(outlier_mask, "OUTLIERS")
    snr_nor, err_nor = summarize(normal_mask,  "NORMAL  ")

    return snr_out, err_out, snr_nor, err_nor, sigma


# ─────────────────────────────────────────────
# STEP 4 — VISUALIZE COMPARISON
# ─────────────────────────────────────────────

def step4_visualize(results_toy, results_real):
    print("\n" + "="*60)
    print("STEP 4 — VISUALIZATION")
    print("="*60)

    snr_out_toy, err_out_toy, snr_nor_toy, err_nor_toy, sigma_toy = results_toy
    snr_out_rl,  err_out_rl,  snr_nor_rl,  err_nor_rl,  sigma_rl  = results_real

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Noise Damage: Toy Graph vs Real-Scale Graph\n"
                 "(Uniform noise calibrated to global max degree)",
                 fontsize=13, fontweight='bold')

    # ── Plot helper ──
    def plot_snr(ax, snr_out, snr_nor, sigma, title):
        ax.axvline(x=1, color='red', linewidth=2,
                   linestyle='--', label='SNR=1 threshold (destroyed below)')
        if len(snr_out) > 0:
            ax.hist(snr_out, bins=15, alpha=0.7,
                    color='orange', label=f'Outliers (n={len(snr_out)})', edgecolor='black')
        ax.hist(snr_nor, bins=15, alpha=0.7,
                color='steelblue', label=f'Normal (n={len(snr_nor)})', edgecolor='black')
        ax.set_xlabel('SNR (signal-to-noise ratio)')
        ax.set_ylabel('Number of nodes')
        ax.set_title(f'{title}\nσ = {sigma:.3f}')
        ax.legend()

    def plot_error(ax, err_out, err_nor, title):
        data   = []
        labels = []
        colors = []
        if len(err_out) > 0:
            data.append(err_out);   labels.append('Outliers'); colors.append('orange')
        data.append(err_nor);       labels.append('Normal');   colors.append('steelblue')
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Error % = ||noisy - original|| / ||original|| × 100')
        ax.set_title(f'{title}\nError distribution per node type')
        ax.axhline(y=100, color='red', linestyle='--',
                   linewidth=1.5, label='100% error line')
        ax.legend()

    plot_snr(  axes[0,0], snr_out_toy, snr_nor_toy, sigma_toy, "TOY GRAPH — SNR")
    plot_error(axes[0,1], err_out_toy, err_nor_toy,             "TOY GRAPH — Error %")
    plot_snr(  axes[1,0], snr_out_rl,  snr_nor_rl,  sigma_rl,  "REAL GRAPH — SNR")
    plot_error(axes[1,1], err_out_rl,  err_nor_rl,             "REAL GRAPH — Error %")

    plt.tight_layout()
    plt.savefig("noise_damage_comparison.png", dpi=150, bbox_inches='tight')
    print("\n  Saved: noise_damage_comparison.png")
    plt.show()


# ─────────────────────────────────────────────
# MAIN — RUN ALL STEPS
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Step 1 — toy graph
    G_toy, feat_toy, types_toy = step1_build_toy_graph()

    # Step 2 — real-scale graph
    G_real, feat_real, types_real, deg_real = step2_build_real_graph()

    # Step 3 — apply noise and measure damage on both
    results_toy  = step3_apply_noise_and_measure(
                       G_toy,  feat_toy,  types_toy,  "TOY GRAPH",  epsilon=1.0)
    results_real = step3_apply_noise_and_measure(
                       G_real, feat_real, types_real, "REAL GRAPH", epsilon=1.0)

    # Step 4 — visualize
    step4_visualize(results_toy, results_real)
