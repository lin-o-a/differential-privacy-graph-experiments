import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Parameters ──────────────────────────────────────────────
n = 1000
s = 5
epsilon = 1.0
delta = 1e-5
np.random.seed(42)

# ── Graphs ───────────────────────────────────────────────────
G_er = nx.erdos_renyi_graph(n, p=0.02)
G_pl = nx.barabasi_albert_graph(n, m=2)


# ── LNDP* Core ───────────────────────────────────────────────
def lndp_estimate(G, s, epsilon, delta):
    degrees = np.array([d for _, d in G.degree()])
    n = len(degrees)
    D = int(degrees.max())
    num_b = D // s + 2
    bucket_v = np.arange(num_b) * s

    agg = np.zeros(num_b)
    for d in degrees:
        low = int(d) // s
        frac = (d % s) / s
        high = low + 1
        agg[low] += (1.0 - frac)
        if high < num_b:
            agg[high] += frac

    true_blurry = agg / n
    sensitivity = 2.0 * np.sqrt(2) / s
    sigma = sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon
    noisy_blurry = true_blurry + np.random.normal(0, sigma / n, num_b)
    noisy_blurry = np.clip(noisy_blurry, 0, None)
    if noisy_blurry.sum() > 0:
        noisy_blurry /= noisy_blurry.sum()

    return true_blurry, noisy_blurry, bucket_v, degrees


# ── Total Variation Distance ─────────────────────────────────
def tvd(p, q):
    L = max(len(p), len(q))
    pp = np.pad(p, (0, L - len(p)))
    qq = np.pad(q, (0, L - len(q)))
    return 0.5 * np.sum(np.abs(pp - qq))


# ── Degree Sequence Reconstruction (Erdős–Gallai) ────────────
def reconstruct_graph_from_noisy_estimate(noisy_dist, buckets, n):
    raw_counts = np.round(noisy_dist * n).astype(int)
    degree_seq = []
    for count, bucket in zip(raw_counts, buckets):
        degree_seq.extend([int(bucket)] * max(0, count))

    degree_seq = sorted(degree_seq, reverse=True)[:n]
    while len(degree_seq) < n:
        degree_seq.append(0)

    degree_seq = np.array(degree_seq, dtype=int)
    degree_seq = np.clip(degree_seq, 0, n - 1)

    if degree_seq.sum() % 2 != 0:
        degree_seq[0] = max(0, degree_seq[0] - 1)

    try:
        G_recon = nx.configuration_model(degree_seq.tolist())
        G_recon = nx.Graph(G_recon)
        G_recon.remove_edges_from(nx.selfloop_edges(G_recon))
    except Exception:
        G_recon = nx.empty_graph(n)

    return G_recon, degree_seq


# ── Run both graphs ──────────────────────────────────────────
results = {}
for G, name in [(G_er, 'Erdős–Rényi'), (G_pl, 'Scale-Free')]:
    tb, nb, bv, degs = lndp_estimate(G, s, epsilon, delta)

    bin_edges = np.arange(0, int(degs.max()) + s + 1, s)
    true_hist, _ = np.histogram(degs, bins=bin_edges, density=False)
    true_hist = true_hist / true_hist.sum()

    tv = tvd(nb, tb)

    G_recon, recon_seq = reconstruct_graph_from_noisy_estimate(nb, bv, n)
    recon_degs = np.array([d for _, d in G_recon.degree()])

    results[name] = {
        'G': G, 'true_blurry': tb, 'noisy_blurry': nb,
        'buckets': bv, 'true_hist': true_hist,
        'true_degs': degs, 'tvd': tv,
        'G_recon': G_recon, 'recon_degs': recon_degs,
        'bin_edges': bin_edges
    }

# ── Plot ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11))
gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

titles = {
    'Erdős–Rényi': 'Concentrated Topology (Erdős–Rényi)\n(Standard Theoretical Assumption)',
    'Scale-Free': 'Heavy-Tailed Topology (Scale-Free)\n(Common Real-World Structure)'
}

for col, name in enumerate(['Erdős–Rényi', 'Scale-Free']):
    r = results[name]
    bv = r['buckets']
    w = s * 0.3

    bg_color = '#F4FFF4' if col == 0 else '#FFF4F4'

    # ── Row 0: LNDP* estimate vs truth ──────────────────────
    ax0 = fig.add_subplot(gs[0, col])
    ax0.set_facecolor(bg_color)

    ax0.bar(bv, r['true_blurry'], width=w,
            color='steelblue', alpha=0.85, label='True blurry')
    ax0.bar(bv + w, r['noisy_blurry'], width=w,
            color='salmon', alpha=0.85, label='LNDP* noisy estimate')
    ax0.bar(r['bin_edges'][:-1] + w * 2,
            r['true_hist'][:len(bv)], width=w * 0.7,
            color='navy', alpha=0.45, label='True exact')

    xlim = min(r['true_degs'].max() + 2 * s, 110)
    ax0.set_xlim(-s, xlim)
    ax0.set_title(titles[name], fontsize=13, fontweight='bold')
    ax0.set_xlabel('Degree bucket (Number of edges per user)', fontweight='bold')
    ax0.set_ylabel('Fraction of users in network')

    ax0.legend(fontsize=9, loc='upper left' if col == 0 else 'center right')
    ax0.axhline(0, color='black', linewidth=0.5)

    color = 'green' if r['tvd'] < 0.1 else 'red'
    ax0.text(0.97, 0.95,
             f"Privacy Error (TVD) = {r['tvd']:.3f}  {'✓ OK' if r['tvd'] < 0.1 else '✗ FAIL'}",
             transform=ax0.transAxes, ha='right', va='top',
             fontsize=11, color=color,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    if col == 1:
        orig_max_hub = int(r['true_degs'].max())

        ax0.annotate(
            'Standard TVD is heavily weighted by the 80% majority.\nWhile the overall score is excellent, it mathematically\nobscures the exposure of extreme outliers in the tail.',
            xy=(2, 0.75), xytext=(20, 0.60),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            fontsize=10, fontweight='bold', color='black',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))

        ax0.plot(orig_max_hub, 0.02, marker='*', color='blue', markersize=15)
        ax0.text(orig_max_hub, 0.05, 'True\nOutlier Node\n(e.g. Hospital)', ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color='blue')

    # ── Row 1: Reconstruction Attack ────────────────────────
    ax1 = fig.add_subplot(gs[1, col])
    ax1.set_facecolor(bg_color)

    max_d = max(r['true_degs'].max(), r['recon_degs'].max() if len(r['recon_degs']) else 1)
    bins = np.arange(0, int(max_d) + s + 1, s)

    th, _ = np.histogram(r['true_degs'], bins=bins, density=False)
    rh, _ = np.histogram(r['recon_degs'], bins=bins, density=False)
    th = th / th.sum() if th.sum() > 0 else th
    rh = rh / rh.sum() if rh.sum() > 0 else rh

    bcentres = bins[:-1]
    ax1.bar(bcentres, th, width=w * 1.1,
            color='steelblue', alpha=0.7, label='Original graph degrees')
    ax1.bar(bcentres + w, rh, width=w * 1.1,
            color='darkorange', alpha=0.7, label='Reconstructed graph degrees')

    ax1.set_xlim(-s, min(int(max_d) + 2 * s, 110))
    ax1.set_title(f'Reconstruction Attack — {name.split()[0]}', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Degree bucket (Number of edges per user)', fontweight='bold')
    ax1.set_ylabel('Fraction of users in network')

    ax1.legend(fontsize=9, loc='upper left')

    recon_tvd = tvd(th[:len(rh)], rh[:len(th)])
    orig_max_hub = int(r['true_degs'].max())
    recon_max_hub = int(r['recon_degs'].max()) if len(r['recon_degs']) > 0 else 0

    hub_text = (f"Recon Error = {recon_tvd:.3f}\n"
                f"Original Max Hub: {orig_max_hub} edges\n"
                f"Reconstructed Max Hub: {recon_max_hub} edges")

    ax1.text(0.97, 0.95, hub_text,
             transform=ax1.transAxes, ha='right', va='top',
             fontsize=10, color='black',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='darkorange', alpha=0.9))

    # Explaining k-anonymity vs Structural Leak (DYNAMIC TEXT)
    if col == 0:
        # Find the y-value of the reconstructed max hub bar to point the arrow correctly
        y_val = rh[-1] if len(rh) > 0 and rh[-1] > 0 else 0.02
        ax1.annotate(
            f'Natural k-anonymity:\nThe reconstructed max hub ({recon_max_hub})\nhides safely in the dense crowd.',
            xy=(recon_max_hub, y_val), xytext=(recon_max_hub, 0.20),
            arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=8),
            fontsize=10, fontweight='bold', color='darkgreen', ha='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.9))
    else:
        ax1.plot(recon_max_hub, 0.02, marker='*', color='red', markersize=18, markeredgecolor='black')
        ax1.annotate(
            f'Structural Leak:\nReconstructing a {recon_max_hub}-edge hub when most nodes have < 5 edges\nperfectly isolates the target. Exact precision ({orig_max_hub})\nis not needed for a successful privacy attack.',
            xy=(recon_max_hub, 0.04), xytext=(25, 0.25),
            arrowprops=dict(facecolor='darkred', shrink=0.05, width=2, headwidth=10),
            fontsize=10, fontweight='bold', color='darkred',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='darkred', alpha=0.9))

fig.suptitle(
    f'Applied Constraints Analysis of LNDP*  (ε={epsilon}, s={s}, n={n})\n'
    'TOP: The Privacy Score Trap  |  '
    'BOTTOM: Adversary Reconstruction Attack from Stored Aggregate',
    fontsize=16, fontweight='bold', y=0.98
)

plt.savefig('lndp_presentation_final_v5.png', dpi=150, bbox_inches='tight')
plt.show()
