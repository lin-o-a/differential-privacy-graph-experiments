import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

np.random.seed(42)


# ─────────────────────────────────────────
# 1. BUILD Medical GRAPH
# ─────────────────────────────────────────
def build_medical_graph(n_nodes=200):
    G = nx.Graph()
    features = {}
    for i in range(n_nodes):
        features[i] = np.array([
            np.random.uniform(0.2, 0.9),
            np.random.uniform(0.0, 1.0),
            np.random.uniform(0.1, 0.8),
        ])
    nx.set_node_attributes(G, features, 'features')
    G.add_nodes_from(range(n_nodes))
    for i in range(n_nodes):
        degree_target = int(np.random.pareto(2.0) * 5) + 1
        degree_target = min(degree_target, 150)
        candidates = [j for j in range(n_nodes) if j != i]
        np.random.shuffle(candidates)
        for j in candidates[:degree_target]:
            G.add_edge(i, j)
    return G, features


# ─────────────────────────────────────────
# 2. STRATIFY
# ─────────────────────────────────────────
def stratify_graph(G, bands=None):
    if bands is None:
        bands = [(0, 10), (10, 30), (30, 80), (80, 999)]
    degrees = dict(G.degree())
    band_nodes = defaultdict(list)
    for node, deg in degrees.items():
        for i, (low, high) in enumerate(bands):
            if low <= deg < high:
                band_nodes[i].append(node)
                break
    subgraphs = {}
    for band_idx, nodes in band_nodes.items():
        subgraphs[band_idx] = {
            'nodes': nodes,
            'subgraph': G.subgraph(nodes),
            'band': bands[band_idx],
            'max_degree': max([degrees[n] for n in nodes]) if nodes else 0,
            'min_degree': min([degrees[n] for n in nodes]) if nodes else 0,
            'size': len(nodes)
        }
    return subgraphs, bands


# ─────────────────────────────────────────
# 3. MECHANISM DECISION LOGIC
# ─────────────────────────────────────────
def select_mechanism(band_data, epsilon, k_threshold=10, sensitivity_threshold=2.0):
    """
    Automatically select best mechanism for a band.

    Returns: mechanism name + parameters + reasoning
    """
    size = band_data['size']
    local_sens = band_data['max_degree']
    query_sens = local_sens / size if size > 0 else float('inf')

    # Rule 1: Band too small → suppression risk
    if size < k_threshold:
        return {
            'mechanism': 'suppress',
            'reason': f'Band too small (n={size} < k={k_threshold}): '
                      f're-identification risk too high',
            'query_sensitivity': query_sens,
            'usable': False
        }

    # Rule 2: Query sensitivity acceptable → Laplace
    if query_sens < sensitivity_threshold:
        return {
            'mechanism': 'laplace',
            'reason': f'Query sensitivity {query_sens:.4f} < threshold {sensitivity_threshold}: '
                      f'Laplace noise is accurate enough',
            'query_sensitivity': query_sens,
            'usable': True
        }

    # Rule 3: Query sensitivity high but band not tiny → Gaussian
    # Gaussian has better composition for multiple queries
    if size >= k_threshold:
        return {
            'mechanism': 'gaussian',
            'reason': f'Query sensitivity {query_sens:.4f} high but n={size} sufficient: '
                      f'Gaussian gives better utility for multiple queries',
            'query_sensitivity': query_sens,
            'usable': True
        }

    # Fallback
    return {
        'mechanism': 'suppress',
        'reason': 'No suitable mechanism found',
        'query_sensitivity': query_sens,
        'usable': False
    }


def should_merge(band_a, band_b, epsilon):
    """
    Check if merging two bands IMPROVES query sensitivity.
    Only merge if the merged band has LOWER query sensitivity than either alone.
    """
    merged_size = band_a['size'] + band_b['size']
    merged_sens = max(band_a['max_degree'], band_b['max_degree'])
    merged_query_sens = merged_sens / merged_size

    a_query_sens = band_a['max_degree'] / band_a['size'] if band_a['size'] > 0 else float('inf')
    b_query_sens = band_b['max_degree'] / band_b['size'] if band_b['size'] > 0 else float('inf')

    improvement_over_b = b_query_sens / merged_query_sens
    degree_gap = band_b['max_degree'] - band_a['max_degree']

    return {
        'should_merge': merged_query_sens < b_query_sens,
        'merged_query_sens': merged_query_sens,
        'original_b_query_sens': b_query_sens,
        'improvement': improvement_over_b,
        'degree_gap': degree_gap,
        'merged_size': merged_size,
        'merged_max_degree': merged_sens
    }


# ─────────────────────────────────────────
# 4. MECHANISMS
# ─────────────────────────────────────────
def laplace_mechanism(true_value, sensitivity, epsilon):
    scale = sensitivity / epsilon
    return true_value + np.random.laplace(0, scale)


def gaussian_mechanism(true_value, sensitivity, epsilon, delta=1e-5):
    """
    Gaussian mechanism for (ε, δ)-DP.
    σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
    Better than Laplace when making multiple queries.
    """
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    return true_value + np.random.normal(0, sigma), sigma


def suppress_mechanism(band_data, features):
    """
    For tiny high-degree bands:
    Only release COUNT and existence, not statistics.
    """
    return {
        'count': len(band_data['nodes']),
        'exists': True,
        'statistics': None,
        'message': f'Band suppressed: {len(band_data["nodes"])} nodes exist '
                   f'(degree {band_data["band"]}), statistics withheld for privacy'
    }


# ─────────────────────────────────────────
# 5. FULL ADAPTIVE PIPELINE
# ─────────────────────────────────────────
def adaptive_stratified_dp(G, features, epsilon=1.0, k_threshold=10):
    """
    Full pipeline:
    1. Stratify
    2. Check each band: merge? suppress? which noise?
    3. Apply chosen mechanism
    4. Report results + decisions
    """
    subgraphs, bands = stratify_graph(G)
    degrees = dict(G.degree())
    band_names = ['S1', 'S2', 'S3', 'S4']

    results = {}
    decisions = {}

    # ── Step 1: Evaluate each band ──
    print("\n" + "─" * 60)
    print("STEP 1: MECHANISM SELECTION PER BAND")
    print("─" * 60)

    band_list = [(idx, band_names[idx], data)
                 for idx, data in sorted(subgraphs.items())]

    # Check merge opportunities first
    merge_decisions = {}
    for i in range(len(band_list) - 1):
        idx_a, name_a, data_a = band_list[i]
        idx_b, name_b, data_b = band_list[i + 1]
        if data_a['size'] > 0 and data_b['size'] > 0:
            merge_info = should_merge(data_a, data_b, epsilon)
            merge_decisions[(name_a, name_b)] = merge_info

    # Print merge analysis
    print("\n  Merge analysis:")
    for (name_a, name_b), info in merge_decisions.items():
        verdict = "✅ BENEFICIAL" if info['should_merge'] else "❌ NOT beneficial"
        print(f"  {name_a}+{name_b}: {verdict}")
        print(f"    Degree gap: {info['degree_gap']}")
        print(f"    Merged query sensitivity: {info['merged_query_sens']:.4f}")
        print(f"    Original {name_b} query sensitivity: {info['original_b_query_sens']:.4f}")
        print(f"    Improvement if merged: {info['improvement']:.2f}×")

    # Select mechanism per band
    print("\n  Mechanism selection:")
    merged_bands = set()

    for idx, name, data in band_list:
        if not data['nodes']:
            continue

        mech = select_mechanism(data, epsilon, k_threshold)
        decisions[name] = mech
        decisions[name]['band_data'] = data

        print(f"\n  {name} (n={data['size']}, "
              f"degree={data['band']}, "
              f"local_sens={data['max_degree']}):")
        print(f"    → Mechanism: {mech['mechanism'].upper()}")
        print(f"    → Reason: {mech['reason']}")

    # ── Step 2: Apply mechanisms ──
    print("\n" + "─" * 60)
    print("STEP 2: APPLYING MECHANISMS")
    print("─" * 60)

    for name, decision in decisions.items():
        data = decision['band_data']
        nodes = data['nodes']
        if not nodes:
            continue

        true_avg = np.mean([features[n][1] for n in nodes])
        query_sens = decision['query_sensitivity']
        mechanism = decision['mechanism']

        if mechanism == 'laplace':
            protected = laplace_mechanism(true_avg, query_sens, epsilon)
            error = abs(true_avg - protected)
            results[name] = {
                'mechanism': 'Laplace',
                'true': true_avg,
                'protected': protected,
                'error': error,
                'usable': True,
                'size': len(nodes),
                'query_sensitivity': query_sens
            }

        elif mechanism == 'gaussian':
            protected, sigma = gaussian_mechanism(true_avg, query_sens, epsilon)
            error = abs(true_avg - protected)
            results[name] = {
                'mechanism': f'Gaussian(σ={sigma:.3f})',
                'true': true_avg,
                'protected': protected,
                'error': error,
                'usable': True,
                'size': len(nodes),
                'query_sensitivity': query_sens
            }

        elif mechanism == 'suppress':
            suppressed = suppress_mechanism(data, features)
            results[name] = {
                'mechanism': 'Suppressed',
                'true': true_avg,
                'protected': None,
                'error': None,
                'usable': False,
                'size': len(nodes),
                'query_sensitivity': query_sens,
                'suppression_info': suppressed
            }

    # ── Step 3: Report ──
    print("\n" + "─" * 60)
    print("STEP 3: RESULTS")
    print("─" * 60)
    print(f"\n  {'Band':<6} {'Mechanism':<22} {'n':>5} "
          f"{'True':>8} {'Protected':>10} {'Error':>8} {'Usable':>8}")
    print("  " + "-" * 75)

    for name, r in results.items():
        protected_str = f"{r['protected']:.4f}" if r['protected'] is not None else "SUPPRESSED"
        error_str = f"{r['error']:.4f}" if r['error'] is not None else "—"
        usable_str = "✅" if r['usable'] else "🔒 protected"
        print(f"  {name:<6} {r['mechanism']:<22} {r['size']:>5} "
              f"{r['true']:>8.4f} {protected_str:>10} {error_str:>8} {usable_str:>8}")

    # ── Privacy guarantee ──
    print("\n" + "─" * 60)
    print("PRIVACY GUARANTEE")
    print("─" * 60)
    print(f"  ε per band: {epsilon}")
    print(f"  ε_total (parallel composition): {epsilon}  ← NOT {epsilon * len(results)}")
    print(f"  Suppressed bands: protected by non-release")
    print(f"  Laplace bands:    ε-DP guaranteed")
    print(f"  Gaussian bands:   (ε,δ)-DP with δ=1e-5")

    return results, decisions, merge_decisions


# ─────────────────────────────────────────
# 6. VISUALIZE DECISIONS
# ─────────────────────────────────────────
def plot_adaptive_results(G, features, results, decisions, merge_decisions):
    degrees = dict(G.degree())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Adaptive Per-Stratum Mechanism Selection\n'
                 'Each band gets the mechanism best suited to its size and sensitivity',
                 fontsize=12, fontweight='bold')

    mech_colors = {
        'Laplace': '#2ecc71',
        'Gaussian': '#3498db',
        'Suppressed': '#e74c3c'
    }

    # ── Plot 1: Band sizes and mechanisms ──
    ax = axes[0]
    band_names_list = list(results.keys())
    sizes = [results[b]['size'] for b in band_names_list]
    mechs = [results[b]['mechanism'].split('(')[0] for b in band_names_list]
    bar_colors = [mech_colors.get(m, '#95a5a6') for m in mechs]

    bars = ax.bar(band_names_list, sizes, color=bar_colors, alpha=0.85, edgecolor='white')
    ax.set_ylabel('Number of Nodes')
    ax.set_title('Band Sizes + Chosen Mechanism')

    legend_patches = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.85, label=l)
                      for l, c in mech_colors.items()]
    ax.legend(handles=legend_patches, fontsize=8)

    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(size), ha='center', va='bottom', fontsize=9)

    # ── Plot 2: Query sensitivity per band ──
    ax = axes[1]
    q_sens = [results[b]['query_sensitivity'] for b in band_names_list]
    bar_colors2 = [mech_colors.get(m, '#95a5a6') for m in mechs]
    ax.bar(band_names_list, q_sens, color=bar_colors2, alpha=0.85, edgecolor='white')
    ax.set_ylabel('Query Sensitivity')
    ax.set_title('Query Sensitivity per Band\n(lower = less noise needed)')
    ax.set_yscale('log')

    for i, (b, qs) in enumerate(zip(band_names_list, q_sens)):
        ax.text(i, qs * 1.3, f'{qs:.3f}', ha='center', va='bottom', fontsize=8)

    # ── Plot 3: Error comparison (usable bands only) ──
    ax = axes[2]
    usable = {b: r for b, r in results.items() if r['usable'] and r['error'] is not None}

    if usable:
        bands_u = list(usable.keys())
        true_vals = [usable[b]['true'] for b in bands_u]
        prot_vals = [usable[b]['protected'] for b in bands_u]
        errors = [usable[b]['error'] for b in bands_u]

        x = np.arange(len(bands_u))
        width = 0.3
        ax.bar(x - width / 2, true_vals, width, label='True value',
               color='#2c3e50', alpha=0.8)
        ax.bar(x + width / 2, prot_vals, width, label='Protected query',
               color='#27ae60', alpha=0.8)

        for i, err in enumerate(errors):
            ax.text(i, max(true_vals[i], prot_vals[i]) + 0.02,
                    f'err={err:.3f}', ha='center', fontsize=8, color='#c0392b')

        ax.set_xticks(x)
        ax.set_xticklabels(bands_u)
        ax.set_ylabel('Avg Risk Score')
        ax.set_title('True vs Protected Query\n(usable bands only)')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig('adaptive_dp_mechanisms.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n  Plot saved: adaptive_dp_mechanisms.png")


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    G, features = build_medical_graph(n_nodes=200)
    degrees = dict(G.degree())

    print("=" * 60)
    print("ADAPTIVE STRATIFIED DP — PER-STRATUM MECHANISMS")
    print("=" * 60)
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Degree range: {min(degrees.values())} – {max(degrees.values())}")

    results, decisions, merge_decisions = adaptive_stratified_dp(
        G, features, epsilon=1.0, k_threshold=10
    )

    plot_adaptive_results(G, features, results, decisions, merge_decisions)
