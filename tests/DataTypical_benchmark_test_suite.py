"""
DataTypical v0.7 - Benchmark Suite
===================================
Tests all three data modalities with realistic datasets.
Validates fast_mode and shapley_compute_formative behavior.

Benchmarks:
1. TABULAR: Small (fast mode), Medium (publication mode), Large (subsampling)
2. TEXT: Keywords, Metadata-based stereotypes
3. GRAPH: Protein network, Molecular network with topology features
4. MODE COMPARISON: Fast vs Publication performance

Author: Amanda S. Barnard
"""

import sys
import os
# Make the local DataTypical source (the parent of this tests/ directory)
# take import priority over any installed copy, so the suite always tests
# this working tree regardless of what is pip-installed.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

from datatypical import DataTypical

print("="*80)
print("DataTypical v0.7 - Benchmark Suite")
print("="*80)

# ============================================================================
# Benchmark Results Tracker
# ============================================================================
benchmark_results = {}

def run_benchmark(name, func):
    """Run a benchmark and track results."""
    print(f"\n{'-'*80}")
    print(f"BENCHMARK: {name}")
    print("-"*80)
    
    try:
        start = time.time()
        result = func()
        elapsed = time.time() - start
        
        benchmark_results[name] = {
            'status': 'PASS',
            'time': elapsed,
            'error': None,
            'result': result
        }
        
        print(f"\nPASSED in {elapsed:.2f}s")
        return True
        
    except Exception as e:
        elapsed = time.time() - start
        benchmark_results[name] = {
            'status': 'FAIL',
            'time': elapsed,
            'error': str(e),
            'result': None
        }
        
        print(f"\nFAILED after {elapsed:.2f}s")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# BENCHMARK 1: Small Tabular Dataset (Fast Mode)
# ============================================================================
def benchmark_small_tabular():
    """Small tabular dataset with fast_mode (explanations only)."""
    np.random.seed(42)
    
    # Generate synthetic molecular data
    n = 200
    data = pd.DataFrame({
        'compound_id': [f'CMPD_{i:04d}' for i in range(n)],
        'mol_weight': np.random.lognormal(5.3, 0.3, n),
        'logP': np.random.normal(2.5, 1.2, n),
        'num_H_donors': np.random.poisson(2, n),
        'num_H_acceptors': np.random.poisson(3, n),
        'TPSA': np.random.gamma(3, 20, n),
        'num_rotatable_bonds': np.random.poisson(5, n),
        'activity': np.random.lognormal(1, 1.5, n)
    })
    
    print(f"Dataset: {len(data)} compounds, 7 features")
    print(f"Mode: fast_mode=True (explanations, no formative)")
    
    # Test with Shapley in fast mode
    dt = DataTypical(
        label_columns=['compound_id'],
        stereotype_column='activity',
        stereotype_target='max',
        shapley_mode=True,
        fast_mode=True,  # Skips formative, computes explanations
        random_state=42,
        verbose=False
    )
    
    results = dt.fit_transform(data)
    
    # Validations
    assert len(results) == n, "Row count mismatch"
    assert 'archetypal_rank' in results.columns, "Missing archetypal_rank"
    assert 'archetypal_shapley_rank' in results.columns, "Missing Shapley rank columns"
    
    # Fast mode: formative should be None
    assert dt.shapley_compute_formative == False, "Fast mode should skip formative"
    assert results['archetypal_shapley_rank'].isna().all(), "Fast mode should have None formative values"
    
    # But explanations should work
    top_sample = results['archetypal_rank'].idxmax()
    explanations = dt.get_shapley_explanations(top_sample)
    assert 'archetypal' in explanations, "Missing archetypal explanations"
    assert explanations['archetypal'].sum() != 0, "Explanations should have values"
    
    print(f"  [OK] Top archetypal: {results['archetypal_rank'].max():.4f}")
    print(f"  [OK] Explanations working (sum: {explanations['archetypal'].sum():.4f})")
    print(f"  [OK] Formative correctly skipped (all None)")
    
    return results


# ============================================================================
# BENCHMARK 2: Medium Tabular Dataset (Publication Mode)
# ============================================================================
def benchmark_medium_tabular_publication():
    """Medium tabular with publication mode (full dual-perspective)."""
    np.random.seed(42)
    
    # Generate synthetic drug discovery data
    n = 300  # Reduced from 500 for faster testing
    data = pd.DataFrame({
        'id': [f'MOL_{i:05d}' for i in range(n)],
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n),
        'feature_3': np.random.randn(n),
        'feature_4': np.random.randn(n),
        'target': np.random.randn(n)
    })
    
    print(f"Dataset: {len(data)} molecules, 4 features")
    print(f"Mode: fast_mode=False (full dual-perspective)")
    
    # Test with publication mode
    dt = DataTypical(
        label_columns=['id'],
        stereotype_column='target',
        stereotype_target='min',
        shapley_mode=True,
        fast_mode=False,  # Computes both formative and explanations
        shapley_n_permutations=30,  # Reduced for faster testing
        random_state=42,
        verbose=False
    )
    
    results = dt.fit_transform(data)
    
    # Validations
    assert len(results) == n, "Row count mismatch"
    assert 'archetypal_shapley_rank' in results.columns, "Missing Shapley ranks"
    
    # Publication mode: formative should have values
    assert dt.shapley_compute_formative == True, "Publication mode should compute formative"
    assert not results['archetypal_shapley_rank'].isna().all(), "Publication mode should have formative values"
    
    # Check formative attributions work
    top_formative = results['archetypal_shapley_rank'].idxmax()
    attributions = dt.get_formative_attributions(top_formative)
    assert 'archetypal' in attributions, "Missing archetypal attributions"
    
    # Check rankings are properly normalized
    assert results['archetypal_rank'].max() <= 1.0, "Archetypal rank > 1"
    assert results['archetypal_rank'].min() >= 0.0, "Archetypal rank < 0"
    
    print(f"  [OK] Archetypal range: [{results['archetypal_rank'].min():.4f}, {results['archetypal_rank'].max():.4f}]")
    print(f"  [OK] Formative range: [{results['archetypal_shapley_rank'].min():.4f}, {results['archetypal_shapley_rank'].max():.4f}]")
    print(f"  [OK] Formative attributions working")
    
    return results


# ============================================================================
# BENCHMARK 3: Large Tabular Dataset (Fast + Subsampling)
# ============================================================================
def benchmark_large_tabular():
    """Large tabular dataset with subsampling."""
    np.random.seed(42)
    
    n = 1000
    data = pd.DataFrame({
        'sample_id': [f'S_{i:05d}' for i in range(n)],
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'x3': np.random.randn(n),
        'x4': np.random.randn(n),
        'x5': np.random.randn(n),
        'outcome': np.random.rand(n)
    })
    
    print(f"Dataset: {len(data)} samples, 5 features")
    print(f"Mode: fast_mode=True + shapley_top_n=100")
    
    # Use Shapley with subsampling for speed
    dt = DataTypical(
        label_columns=['sample_id'],
        stereotype_column='outcome',
        stereotype_target='max',
        shapley_mode=True,
        shapley_top_n=100,  # Only compute for top 100
        fast_mode=True,
        random_state=42,
        verbose=False
    )
    
    results = dt.fit_transform(data)
    
    # Validations
    assert len(results) == n, "Row count mismatch"
    
    # Check subsampling worked for explanations
    # Count non-zero explanation rows
    nonzero_explanations = 0
    for i in range(min(200, n)):  # Check first 200
        exp = dt.get_shapley_explanations(i)
        if exp['archetypal'].sum() != 0:
            nonzero_explanations += 1
    
    print(f"  [OK] Samples with Shapley explanations (first 200 checked): {nonzero_explanations}")
    assert nonzero_explanations <= 110, f"Too many Shapley values computed: {nonzero_explanations}"
    
    # Formative should be None (fast mode)
    assert results['archetypal_shapley_rank'].isna().all(), "Fast mode formative should be None"
    
    print(f"  [OK] Subsampling working correctly")
    print(f"  [OK] Formative correctly skipped")
    
    return results


# ============================================================================
# BENCHMARK 4: Text Data - Small Corpus
# ============================================================================
def benchmark_text_small():
    """Text data with keyword-based stereotypes."""
    np.random.seed(42)
    
    # Generate synthetic research abstracts
    topics = {
        'protein': ['protein', 'enzyme', 'binding', 'structure', 'amino acid'],
        'nanoparticle': ['nanoparticle', 'nanomaterial', 'surface', 'coating', 'size'],
        'catalyst': ['catalyst', 'reaction', 'selectivity', 'conversion', 'mechanism'],
        'battery': ['battery', 'electrode', 'electrolyte', 'charge', 'capacity']
    }
    
    corpus = []
    for _ in range(100):
        topic = np.random.choice(list(topics.keys()))
        words = topics[topic]
        n_words = np.random.poisson(15)
        doc_words = np.random.choice(words + ['study', 'research', 'analysis', 'method'], n_words)
        corpus.append(' '.join(doc_words))
    
    print(f"Corpus: {len(corpus)} documents")
    print(f"Mode: fast_mode=True, shapley_mode=False")
    
    # Test with keywords
    dt = DataTypical(
        stereotype_keywords=['protein', 'enzyme'],
        nmf_rank=5,
        n_prototypes=15,
        shapley_mode=False,  # Skip Shapley for text (slower on text)
        fast_mode=True,
        random_state=42,
        verbose=False
    )
    
    results = dt.fit_transform(corpus)
    
    # Validations
    assert len(results) == len(corpus), "Row count mismatch"
    assert 'stereotypical_rank' in results.columns, "Missing stereotypical_rank"
    
    # Check that protein docs rank higher
    protein_docs = [i for i, doc in enumerate(corpus) if 'protein' in doc]
    if protein_docs:
        avg_protein_rank = results.loc[protein_docs, 'stereotypical_rank'].mean()
        avg_all_rank = results['stereotypical_rank'].mean()
        print(f"  [OK] Avg protein doc rank: {avg_protein_rank:.4f}")
        print(f"  [OK] Avg all doc rank: {avg_all_rank:.4f}")
        assert avg_protein_rank > avg_all_rank, "Keywords not working"
    
    return results


# ============================================================================
# BENCHMARK 5: Text Data with Metadata
# ============================================================================
def benchmark_text_metadata():
    """Text data with metadata-based stereotypes."""
    np.random.seed(42)
    
    corpus = [f"Document about topic {i % 5} with content words" for i in range(50)]
    
    # Add metadata
    metadata = pd.DataFrame({
        'relevance': np.random.rand(50),
        'quality': np.random.rand(50),
        'topic': [f'topic_{i % 5}' for i in range(50)]
    })
    
    print(f"Corpus: {len(corpus)} documents with metadata")
    print(f"Mode: fast_mode=True, shapley_mode=False")
    
    dt = DataTypical(
        stereotype_column='relevance',
        stereotype_target='max',
        nmf_rank=3,
        n_prototypes=10,
        shapley_mode=False,
        fast_mode=True,
        random_state=42,
        verbose=False
    )
    
    results = dt.fit_transform(corpus, text_metadata=metadata)
    
    # Validations
    assert len(results) == len(corpus), "Row count mismatch"
    
    # Check correlation between relevance and stereotypical rank
    corr = np.corrcoef(metadata['relevance'], results['stereotypical_rank'])[0, 1]
    print(f"  [OK] Relevance vs stereotype rank correlation: {corr:.4f}")
    assert corr > 0.5, f"Weak correlation: {corr}"
    
    return results


# ============================================================================
# BENCHMARK 6: Graph Data - Protein Network
# ============================================================================
def benchmark_graph_protein_network():
    """Graph data with topology features."""
    np.random.seed(42)
    
    n_proteins = 100
    
    # Node features (protein properties)
    node_features = pd.DataFrame({
        'protein_id': [f'PROT_{i:03d}' for i in range(n_proteins)],
        'mol_weight': np.random.lognormal(10, 0.3, n_proteins),
        'hydrophobicity': np.random.randn(n_proteins),
        'charge': np.random.randn(n_proteins),
        'expression': np.random.lognormal(5, 1, n_proteins)
    })
    
    # Generate scale-free network (some hubs)
    n_edges = 200
    hub_proteins = np.random.choice(n_proteins, 15, replace=False)
    
    edges = []
    for _ in range(n_edges):
        if np.random.random() < 0.6:  # 60% connect to hubs
            i = np.random.choice(hub_proteins)
        else:
            i = np.random.randint(n_proteins)
        
        j = np.random.randint(n_proteins)
        if i != j:
            edges.append([i, j])
    
    edges = np.array(edges)
    
    print(f"Network: {n_proteins} proteins, {len(edges)} interactions")
    print(f"Mode: fast_mode=True, topology features enabled")
    
    # Test with topology features
    dt = DataTypical(
        label_columns=['protein_id'],
        graph_topology_features=['degree', 'clustering'],
        stereotype_column='expression',
        stereotype_target='max',
        shapley_mode=False,
        fast_mode=True,
        random_state=42,
        verbose=False
    )
    
    results = dt.fit_transform(node_features, edges=edges)
    
    # Validations
    assert len(results) == n_proteins, "Row count mismatch"
    assert 'degree' in results.columns, "Missing topology feature: degree"
    assert 'clustering' in results.columns, "Missing topology feature: clustering"
    
    # Check that hubs have high degree
    hub_degrees = results.loc[hub_proteins, 'degree'].mean()
    all_degrees = results['degree'].mean()
    print(f"  [OK] Hub avg degree: {hub_degrees:.1f}")
    print(f"  [OK] Overall avg degree: {all_degrees:.1f}")
    assert hub_degrees > all_degrees * 1.5, "Hubs should have higher degree"
    
    return results


# ============================================================================
# BENCHMARK 7: Graph Data - Molecular Network
# ============================================================================
def benchmark_graph_molecular():
    """Graph data - molecular similarity network."""
    np.random.seed(42)
    
    n_molecules = 80
    
    # Node features
    node_features = pd.DataFrame({
        'mol_id': [f'MOL_{i:03d}' for i in range(n_molecules)],
        'feature_A': np.random.randn(n_molecules),
        'feature_B': np.random.randn(n_molecules),
        'feature_C': np.random.randn(n_molecules),
        'activity': np.random.rand(n_molecules)
    })
    
    # Random graph
    edges = []
    for i in range(n_molecules):
        n_neighbors = np.random.poisson(3)
        for _ in range(n_neighbors):
            j = np.random.randint(n_molecules)
            if i != j:
                edges.append([i, j])
    
    edges = np.array(edges)
    
    print(f"Network: {n_molecules} molecules, {len(edges)} edges")
    print(f"Mode: fast_mode=True")
    
    dt = DataTypical(
        label_columns=['mol_id'],
        graph_topology_features=['degree', 'pagerank'],
        stereotype_column='activity',
        stereotype_target='max',
        shapley_mode=False,
        fast_mode=True,
        random_state=42,
        verbose=False
    )
    
    results = dt.fit_transform(node_features, edges=edges)
    
    # Validations
    assert len(results) == n_molecules, "Row count mismatch"
    assert 'degree' in results.columns, "Missing degree"
    assert 'pagerank' in results.columns, "Missing pagerank"
    
    print(f"  [OK] Degree range: [{results['degree'].min()}, {results['degree'].max()}]")
    print(f"  [OK] PageRank range: [{results['pagerank'].min():.4f}, {results['pagerank'].max():.4f}]")
    
    return results


# ============================================================================
# BENCHMARK 8: Mode Comparison (Fast vs Publication)
# ============================================================================
def benchmark_mode_comparison():
    """Compare fast_mode=True vs False performance and outputs."""
    np.random.seed(42)
    
    n = 200
    data = pd.DataFrame({
        'id': [f'S{i:03d}' for i in range(n)],
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'x3': np.random.randn(n),
        'target': np.random.rand(n)
    })
    
    print(f"Dataset: {n} samples, 3 features")
    print(f"Comparing fast_mode=True vs False")
    
    # Fast mode
    print("\n  [A] Fast mode...")
    start = time.time()
    dt_fast = DataTypical(
        label_columns=['id'],
        stereotype_column='target',
        stereotype_target='max',
        shapley_mode=True,
        fast_mode=True,
        random_state=42,
        verbose=False
    )
    results_fast = dt_fast.fit_transform(data)
    time_fast = time.time() - start
    
    # Publication mode
    print("  [B] Publication mode...")
    start = time.time()
    dt_pub = DataTypical(
        label_columns=['id'],
        stereotype_column='target',
        stereotype_target='max',
        shapley_mode=True,
        fast_mode=False,
        shapley_n_permutations=30,
        random_state=42,
        verbose=False
    )
    results_pub = dt_pub.fit_transform(data)
    time_pub = time.time() - start
    
    # Validations
    print(f"\n  Timing:")
    print(f"    Fast mode: {time_fast:.2f}s")
    print(f"    Publication mode: {time_pub:.2f}s")
    print(f"    Speedup: {time_pub/time_fast:.1f}x")
    
    # Check behavior differences
    print(f"\n  Formative computation:")
    print(f"    Fast mode: {dt_fast.shapley_compute_formative} (skipped)")
    print(f"    Publication mode: {dt_pub.shapley_compute_formative} (computed)")
    
    # Check archetypal methods
    print(f"\n  Archetypal method:")
    print(f"    Fast mode: {dt_fast.archetypal_method}")
    print(f"    Publication mode: {dt_pub.archetypal_method}")
    
    # Check column presence
    for col in ['archetypal_shapley_rank', 'prototypical_shapley_rank', 'stereotypical_shapley_rank']:
        assert col in results_fast.columns, f"Fast mode missing {col}"
        assert col in results_pub.columns, f"Publication mode missing {col}"
    
    # Check formative values
    assert results_fast['archetypal_shapley_rank'].isna().all(), "Fast mode formative should be None"
    assert not results_pub['archetypal_shapley_rank'].isna().all(), "Publication mode formative should have values"
    
    # Check actual rank correlation
    corr = results_fast['archetypal_rank'].corr(results_pub['archetypal_rank'])
    print(f"\n  Actual rank correlation: {corr:.4f}")
    
    # Adjust correlation threshold based on archetypal methods
    method_fast = dt_fast.archetypal_method
    method_pub = dt_pub.archetypal_method
    
    if method_fast == method_pub:
        # Same method: expect high correlation
        threshold = 0.60
        assert corr > threshold, (
            f"Same archetypal method ({method_fast}), ranks should be correlated: {corr:.4f} "
            f"(expected > {threshold})"
        )
    else:
        # Different methods: expect lower correlation
        threshold = 0.20
        assert corr > threshold, (
            f"Different archetypal methods ({method_fast} vs {method_pub}), "
            f"lower correlation expected: {corr:.4f} (threshold: {threshold}). "
            f"This is normal - NMF and AA produce fundamentally different archetypal rankings."
        )
        print(f"  Note: Different methods ({method_fast} vs {method_pub}) -> lower correlation is expected")
    
    print("\n  [OK] Both modes working correctly")
    print("  [OK] Fast mode skips formative as expected")
    print("  [OK] Publication mode computes full dual-perspective")
    
    return {'fast': results_fast, 'pub': results_pub}


# ============================================================================
# RUN ALL BENCHMARKS
# ============================================================================
print("\nRUNNING BENCHMARK SUITE")

# Tabular benchmarks
run_benchmark("1. Small Tabular (Fast Mode)", benchmark_small_tabular)
run_benchmark("2. Medium Tabular (Publication Mode)", benchmark_medium_tabular_publication)
run_benchmark("3. Large Tabular (Subsampling)", benchmark_large_tabular)

# Text benchmarks
run_benchmark("4. Text Small (Keywords)", benchmark_text_small)
run_benchmark("5. Text Metadata", benchmark_text_metadata)

# Graph benchmarks
run_benchmark("6. Graph Protein Network", benchmark_graph_protein_network)
run_benchmark("7. Graph Molecular", benchmark_graph_molecular)

# Mode comparison
run_benchmark("8. Mode Comparison (Fast vs Publication)", benchmark_mode_comparison)


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("BENCHMARK SUMMARY")
print("="*80)

# Count results
total = len(benchmark_results)
passed = sum(1 for r in benchmark_results.values() if r['status'] == 'PASS')
failed = total - passed

print(f"\n[OK] Successful: {passed}/{total}")
print(f"[FAIL] Failed: {failed}/{total}")

# Modality breakdown
modalities = {
    'Tabular': [1, 2, 3],
    'Text': [4, 5],
    'Graph': [6, 7],
    'Mode Comparison': [8]
}

print("\nMODALITY COVERAGE")
for modality, benchmark_nums in modalities.items():
    modality_benchmarks = [f"{i}. " for i in benchmark_nums]
    modality_passed = sum(
        1 for name, result in benchmark_results.items()
        if any(name.startswith(prefix) for prefix in modality_benchmarks)
        and result['status'] == 'PASS'
    )
    modality_total = len(benchmark_nums)
    status = "[OK]" if modality_passed == modality_total else "[FAIL]"
    print(f"  {status} {modality:20s}: {modality_passed}/{modality_total} benchmark(s)")

# Timing breakdown
print("\nTIMING BREAKDOWN")
for name, result in sorted(benchmark_results.items()):
    status_symbol = "[OK]" if result['status'] == 'PASS' else "[FAIL]"
    print(f"  {status_symbol} {name:55s}: {result['time']:6.2f}s")

total_time = sum(r['time'] for r in benchmark_results.values())
print(f"\n  Total time: {total_time:.2f}s")

# Final verdict
print("\n" + "="*80)
if failed == 0:
    print("[OK] ALL BENCHMARKS PASSED")
    print("="*80)
    print("\nDataTypical v0.7 VALIDATED")
    print("  - All modalities working (Tabular, Text, Graph)")
    print("  - Fast mode optimizations functional")
    print("  - Publication mode dual-perspective working")
    print("  - Shapley explanations + formative validated")
    print("  - Auto-detection working as expected")
else:
    print(f"[FAIL] {failed} BENCHMARK(S) FAILED")
    print("="*80)
    print("\nFailed benchmarks:")
    for name, result in benchmark_results.items():
        if result['status'] == 'FAIL':
            print(f"\n  {name}:")
            print(f"    Error: {result['error']}")

print("="*80)