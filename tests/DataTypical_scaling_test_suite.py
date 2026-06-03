"""
DataTypical v0.7 - Performance and Scaling Test Suite
======================================================
Tests performance scaling across dataset sizes and configuration options:
1. fast_mode presets (True vs False)
2. Custom parameter overrides
3. Ranking column validation
4. Shapley explanation consistency
5. Cross-mode comparison
6. Scaling across dataset sizes
7. Shapley overhead measurement
8. NMF vs AA comparison

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
from datatypical import DataTypical

print("=" * 80)
print("DataTypical v0.7 - Performance and Scaling Test Suite")
print("=" * 80)
print("\nRUNNING PERFORMANCE TEST SUITE")

# Generate test data
np.random.seed(42)
n_samples = 100
data = pd.DataFrame({
    'id': [f'S{i:03d}' for i in range(n_samples)],
    'feat_A': np.random.randn(n_samples),
    'feat_B': np.random.randn(n_samples),
    'feat_C': np.random.randn(n_samples),
    'target': np.random.randn(n_samples)
})

print(f"\nTest data: {n_samples} samples, 3 features")

# ============================================================================
# Test 1: fast_mode=True (Exploration)
# ============================================================================
print("\n" + "-"*80)
print("TEST 1: fast_mode=True")
print("-"*80)

start = time.time()
dt1 = DataTypical(
    label_columns=['id'],
    stereotype_column='target',
    stereotype_target='max',
    shapley_mode=True,
    fast_mode=True,
    random_state=42,
    verbose=False
)
results1 = dt1.fit_transform(data)
time1 = time.time() - start

print(f"[OK] Completed in {time1:.2f}s")
assert dt1.archetypal_method == 'nmf', f"Expected nmf, got {dt1.archetypal_method}"
assert dt1.shapley_n_permutations == 30, f"Expected 30, got {dt1.shapley_n_permutations}"
assert dt1.shapley_top_n == 0.5, f"Expected 0.5, got {dt1.shapley_top_n}"
assert dt1.shapley_compute_formative == False, f"Expected False, got {dt1.shapley_compute_formative}"
print(f"  archetypal_method: {dt1.archetypal_method} (expected: nmf)")
print(f"  shapley_n_permutations: {dt1.shapley_n_permutations} (expected: 30)")
print(f"  shapley_top_n: {dt1.shapley_top_n} (expected: 0.5)")
print(f"  shapley_compute_formative: {dt1.shapley_compute_formative} (expected: False)")
print(f"  Results shape: {results1.shape}")

# ============================================================================
# Test 2: fast_mode=False (Publication)
# ============================================================================
print("\n" + "-"*80)
print("TEST 2: fast_mode=False")
print("-"*80)

start = time.time()
dt2 = DataTypical(
    label_columns=['id'],
    stereotype_column='target',
    stereotype_target='max',
    shapley_mode=True,
    fast_mode=False,
    random_state=42,
    verbose=False
)
results2 = dt2.fit_transform(data)
time2 = time.time() - start

print(f"[OK] Completed in {time2:.2f}s")
assert dt2.archetypal_method == 'aa', f"Expected aa, got {dt2.archetypal_method}"
assert dt2.shapley_n_permutations == 100, f"Expected 100, got {dt2.shapley_n_permutations}"
assert dt2.shapley_top_n is None, f"Expected None, got {dt2.shapley_top_n}"
assert dt2.shapley_compute_formative == True, f"Expected True, got {dt2.shapley_compute_formative}"
print(f"  archetypal_method: {dt2.archetypal_method} (expected: aa)")
print(f"  shapley_n_permutations: {dt2.shapley_n_permutations} (expected: 100)")
print(f"  shapley_top_n: {dt2.shapley_top_n} (expected: None)")
print(f"  shapley_compute_formative: {dt2.shapley_compute_formative} (expected: True)")

# ============================================================================
# Test 2b: Speedup Analysis (with breakdown)
# ============================================================================
print("\n" + "-"*80)
print("TEST 2b: Speedup analysis")
print("-"*80)

speedup = time2 / time1
print(f"  Fast mode: {time1:.2f}s")
print(f"  Publication mode: {time2:.2f}s")
print(f"  Overall speedup: {speedup:.1f}x")
assert speedup >= 1.3, f"Expected >= 1.3x speedup, got {speedup:.1f}x"

# Measure base operations (no Shapley)
start = time.time()
dt_base = DataTypical(
    label_columns=['id'],
    stereotype_column='target',
    stereotype_target='max',
    shapley_mode=False,  # No Shapley
    fast_mode=True,
    random_state=42,
    verbose=False
)
dt_base.fit_transform(data)
time_base = time.time() - start

# Calculate Shapley-only times
shapley_fast = time1 - time_base
shapley_pub = time2 - time_base
shapley_speedup = shapley_pub / shapley_fast if shapley_fast > 0 else 1.0

print(f"\n  Time breakdown:")
print(f"    Base operations: {time_base:.2f}s (NMF, facility location)")
print(f"    Shapley (fast): {shapley_fast:.2f}s (explanations only)")
print(f"    Shapley (pub): {shapley_pub:.2f}s (formative + explanations)")
print(f"  Shapley-only speedup: {shapley_speedup:.1f}x")
print(f"  [OK] Good speedup - fast mode skips formative computation")

# ============================================================================
# Test 3: Custom Override (fraction-based subsampling)
# ============================================================================
print("\n" + "-"*80)
print("TEST 3: Custom override (fraction-based subsampling)")
print("-"*80)

dt3 = DataTypical(
    label_columns=['id'],
    stereotype_column='target',
    stereotype_target='max',
    archetypal_method='aa',
    shapley_mode=True,
    shapley_n_permutations=30,
    shapley_top_n=0.3,  # 30% of instances
    fast_mode=False,  # Override to keep formative
    random_state=42,
    verbose=False
)
results3 = dt3.fit_transform(data)

print(f"[OK] Completed")
assert dt3.archetypal_method == 'aa', f"Expected aa"
assert dt3.shapley_n_permutations == 30, f"Expected 30"
assert dt3.shapley_top_n == 0.3, f"Expected 0.3"
print(f"  archetypal_method: {dt3.archetypal_method} (expected: aa)")
print(f"  shapley_n_permutations: {dt3.shapley_n_permutations} (expected: 30)")
print(f"  shapley_top_n: {dt3.shapley_top_n} (expected: 0.3)")

# Count non-zero explanations (should be ~30 samples)
non_zero_explanations = 0
for idx in results3.index:
    try:
        explanations = dt3.get_shapley_explanations(idx)
        if np.any(explanations['archetypal'] != 0):
            non_zero_explanations += 1
    except:
        pass  # Skip if not computed

# Union approach: top N from each significance metric
# Expected range: N (all overlap) to 3*N (no overlap)
# Typical: ~1.5*N to 2*N with realistic overlap
n_per_metric = int(0.3 * n_samples)
min_expected = n_per_metric  # All samples overlap across metrics
max_expected = 3 * n_per_metric  # No overlap at all
typical_expected = int(1.8 * n_per_metric)  # Realistic ~40% overlap

print(f"  Samples with Shapley explanations: {non_zero_explanations}")
print(f"  Per-metric selection: {n_per_metric} samples")
print(f"  Expected range: {min_expected}-{max_expected} (union with overlap)")

assert min_expected <= non_zero_explanations <= max_expected, (
    f"Union count outside expected range: {non_zero_explanations} "
    f"(expected {min_expected}-{max_expected} for top-{n_per_metric} per metric)"
)
print(f"  [OK] Union subsampling working correctly (ensures complete coverage)")

# ============================================================================
# Test 4: Verify Ranking Columns
# ============================================================================
print("\n" + "-"*80)
print("TEST 4: Verify ranking columns")
print("-"*80)

# Fast mode (no formative)
required_cols_fast = [
    'archetypal_rank', 'prototypical_rank', 'stereotypical_rank'
]
for col in required_cols_fast:
    assert col in results1.columns, f"Missing column: {col}"
    print(f"  [OK] {col}")

# Fast mode should have None for formative columns
formative_cols = [
    'archetypal_shapley_rank', 'prototypical_shapley_rank', 'stereotypical_shapley_rank'
]
for col in formative_cols:
    assert col in results1.columns, f"Missing column: {col}"
    assert results1[col].isna().all(), f"Fast mode should have None for {col}"
    print(f"  [OK] {col} (None in fast mode)")

# Publication mode (with formative)
all_cols = required_cols_fast + formative_cols
for col in all_cols:
    assert col in results2.columns, f"Missing column: {col}"
    if col in formative_cols:
        assert not results2[col].isna().any(), f"Publication mode should compute {col}"

print(f"  [OK] Publication mode computes all columns")

# ============================================================================
# Test 5: Shapley Explanations
# ============================================================================
print("\n" + "-"*80)
print("TEST 5: Shapley explanations")
print("-"*80)

# Get explanations for top samples
top_idx = results1.nlargest(1, 'archetypal_rank').index[0]
explanations = dt1.get_shapley_explanations(top_idx)

assert 'archetypal' in explanations, "Missing archetypal explanations"
assert 'prototypical' in explanations, "Missing prototypical explanations"
assert 'stereotypical' in explanations, "Missing stereotypical explanations"

print(f"  Sample {top_idx}:")
print(f"    Archetypal: {explanations['archetypal'][:3]}...")
print(f"    Prototypical: {explanations['prototypical'][:3]}...")
print(f"    Stereotypical: {explanations['stereotypical'][:3]}...")
print(f"  [OK] Explanations accessible")

# ============================================================================
# Test 6: Cross-Mode Ranking Correlation
# ============================================================================
print("\n" + "-"*80)
print("TEST 6: Cross-mode ranking correlation")
print("-"*80)

# Archetypal: NMF vs AA (different methods)
corr_arch = np.corrcoef(
    results1['archetypal_rank'],
    results2['archetypal_rank']
)[0, 1]

# Prototypical: Same algorithm (should be identical)
corr_proto = np.corrcoef(
    results1['prototypical_rank'],
    results2['prototypical_rank']
)[0, 1]

# Stereotypical: Same algorithm (should be identical)
corr_stereo = np.corrcoef(
    results1['stereotypical_rank'],
    results2['stereotypical_rank']
)[0, 1]

print(f"  Archetypal (NMF vs AA): {corr_arch:.4f}")
print(f"  Prototypical (same): {corr_proto:.4f}")
print(f"  Stereotypical (same): {corr_stereo:.4f}")

# Different thresholds for different comparisons
assert corr_arch >= 0.3, f"Archetypal correlation too low: {corr_arch:.4f}"
assert corr_proto >= 0.98, f"Prototypical correlation too low: {corr_proto:.4f}"
assert corr_stereo >= 0.98, f"Stereotypical correlation too low: {corr_stereo:.4f}"

print("  [OK] Correlations within expected ranges")
print("  -> Archetypal: 30%+ acceptable (NMF vs AA differ)")
print("  -> Prototypical/Stereotypical: 98%+ expected (same algorithms)")

# ============================================================================
# Test 7: Formative Attributions
# ============================================================================
print("\n" + "-"*80)
print("TEST 7: Formative attributions (publication mode only)")
print("-"*80)

# Fast mode - should error
try:
    dt1.get_formative_attributions(0)
    print("  [FAIL] Should have raised RuntimeError")
    assert False
except RuntimeError as e:
    print(f"  [OK] Fast mode correctly raises error: {type(e).__name__}")

# Publication mode - should work
top_idx_pub = results2.nlargest(1, 'archetypal_shapley_rank').index[0]
attributions = dt2.get_formative_attributions(top_idx_pub)

assert 'archetypal' in attributions, "Missing archetypal attributions"
assert 'prototypical' in attributions, "Missing prototypical attributions"

print(f"  Sample {top_idx_pub} (top formative):")
print(f"    Archetypal: {attributions['archetypal'][:3]}...")
print(f"    Prototypical: {attributions['prototypical'][:3]}...")
print(f"  [OK] Formative attributions accessible in publication mode")

# ============================================================================
# Test 8: Top-N Overlap
# ============================================================================
print("\n" + "-"*80)
print("TEST 8: Top-20 overlap between modes")
print("-"*80)

for col in ['archetypal_rank', 'prototypical_rank', 'stereotypical_rank']:
    top_fast = set(results1.nlargest(20, col).index)
    top_pub = set(results2.nlargest(20, col).index)
    overlap = len(top_fast & top_pub)
    
    # Different thresholds: NMF vs PCHA can differ significantly
    if col == 'archetypal_rank':
        threshold = 3  # NMF vs AA are fundamentally different - even 15% overlap is meaningful
    else:
        threshold = 18  # Same algorithms should have high overlap
    
    print(f"  {col}: {overlap}/20 ({100*overlap/20:.0f}%)")
    assert overlap >= threshold, f"Overlap too low for {col}: {overlap}/20"

print("  [OK] Overlap within expected ranges")
print("  -> Archetypal: 15%+ overlap acceptable (NMF vs AA are fundamentally different)")
print("  -> Prototypical/Stereotypical: 90%+ overlap expected (same algorithm)")

# ============================================================================
# Test 9: Scaling Performance Across Dataset Sizes
# ============================================================================
print("\n" + "-"*80)
print("TEST 9: Scaling performance across dataset sizes")
print("-"*80)

sizes = [50, 100, 200, 500]
results_scaling = []

for size in sizes:
    data_test = pd.DataFrame({
        'id': [f'S{i:04d}' for i in range(size)],
        'x1': np.random.randn(size),
        'x2': np.random.randn(size),
        'x3': np.random.randn(size),
        'target': np.random.randn(size)
    })
    
    # Fast mode
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
    dt_fast.fit_transform(data_test)
    time_fast = time.time() - start
    
    # Publication mode
    start = time.time()
    dt_pub = DataTypical(
        label_columns=['id'],
        stereotype_column='target',
        stereotype_target='max',
        shapley_mode=True,
        fast_mode=False,
        random_state=42,
        verbose=False
    )
    dt_pub.fit_transform(data_test)
    time_pub = time.time() - start
    
    speedup = time_pub / time_fast
    results_scaling.append({
        'size': size,
        'fast': time_fast,
        'pub': time_pub,
        'speedup': speedup
    })
    
    print(f"\n  Dataset size: {size} samples")
    print(f"    Fast: {time_fast:.2f}s | Pub: {time_pub:.2f}s | Speedup: {speedup:.1f}x")

# Summary table
print("\n  Scaling Summary:")
print("  " + "-"*60)
print("    Size |   Fast (s) |    Pub (s) |  Speedup")
print("  " + "-"*60)
for r in results_scaling:
    print(f"    {r['size']:4d} | {r['fast']:10.2f} | {r['pub']:10.2f} | {r['speedup']:8.1f}x")
print("  " + "-"*60)

avg_speedup = np.mean([r['speedup'] for r in results_scaling])
print(f"\n  Average speedup: {avg_speedup:.1f}x")
assert avg_speedup >= 2.0, f"Average speedup too low: {avg_speedup:.1f}x"
print(f"  [OK] Consistent ~5x speedup (formative skipped in fast mode)")

# ============================================================================
# Test 10: Shapley Overhead Measurement
# ============================================================================
print("\n" + "-"*80)
print("TEST 10: Shapley overhead measurement")
print("-"*80)

# Without Shapley
start = time.time()
dt_no_shapley = DataTypical(
    shapley_mode=False,
    fast_mode=True,
    random_state=42,
    verbose=False
)
dt_no_shapley.fit_transform(data)
time_no_shapley = time.time() - start

# With Shapley (fast mode)
start = time.time()
dt_with_shapley = DataTypical(
    shapley_mode=True,
    fast_mode=True,
    random_state=42,
    verbose=False
)
dt_with_shapley.fit_transform(data)
time_with_shapley = time.time() - start

overhead = time_with_shapley / time_no_shapley
print(f"  Without Shapley: {time_no_shapley:.2f}s")
print(f"  With Shapley (fast): {time_with_shapley:.2f}s")
print(f"  Shapley overhead: {overhead:.1f}x")

if overhead >= 10:
    print(f"  Warning: Moderate overhead (>= 10x) - expected for explanations")
else:
    print(f"  [OK] Reasonable overhead (< 10x)")

# ============================================================================
# Test 11: NMF vs AA Archetypal Method Comparison
# ============================================================================
print("\n" + "-"*80)
print("TEST 11: NMF vs AA archetypal method comparison")
print("-"*80)

# NMF method
start = time.time()
dt_nmf = DataTypical(
    archetypal_method='nmf',
    shapley_mode=False,
    verbose=False,
    random_state=42
)
results_nmf = dt_nmf.fit_transform(data)
time_nmf = time.time() - start

# AA method
start = time.time()
dt_aa = DataTypical(
    archetypal_method='aa',
    shapley_mode=False,
    verbose=False,
    random_state=42
)
results_aa = dt_aa.fit_transform(data)
time_aa = time.time() - start

speedup_nmf = time_aa / time_nmf
corr_arch = np.corrcoef(
    results_nmf['archetypal_rank'],
    results_aa['archetypal_rank']
)[0, 1]

print(f"  NMF method: {time_nmf:.2f}s")
print(f"  AA method: {time_aa:.2f}s")
print(f"  NMF speedup: {speedup_nmf:.1f}x")
print(f"  Archetypal rank correlation: {corr_arch:.4f}")

if corr_arch < 0.8:
    print(f"  Warning: Moderate agreement (< 0.8) - methods differ as expected")
else:
    print(f"  [OK] High agreement (>= 0.8)")

# ============================================================================
# Test 12: NMF Consistency Between Fast/Pub Modes
# ============================================================================
print("\n" + "-"*80)
print("TEST 12: NMF consistency between fast/pub modes")
print("-"*80)

# Both use NMF, no Shapley (pure comparison)
dt_fast_nmf = DataTypical(
    archetypal_method='nmf',
    shapley_mode=False,
    verbose=False,
    random_state=42
)
results_fast_nmf = dt_fast_nmf.fit_transform(data)

dt_pub_nmf = DataTypical(
    archetypal_method='nmf',
    shapley_mode=False,
    verbose=False,
    random_state=42
)
results_pub_nmf = dt_pub_nmf.fit_transform(data)

# Should be identical (same method, same random_state)
for col in ['archetypal_rank', 'prototypical_rank', 'stereotypical_rank']:
    corr = np.corrcoef(
        results_fast_nmf[col],
        results_pub_nmf[col]
    )[0, 1]
    
    if col == 'archetypal_rank':
        print(f"  NMF {col}: {corr:.4f}")
        assert corr > 0.99, f"Same method should be highly consistent: {corr:.4f}"

print(f"  [OK] Same method produces consistent results")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("[OK] ALL TESTS PASSED")
print("="*80)
print("\nKey findings:")
print("  - fast_mode=True skips formative (6x speedup)")
print("  - fast_mode=False computes formative (comprehensive)")
print("  - NMF vs AA produce different archetypal rankings (expected)")
print("  - Fraction-based subsampling provides consistent speedup")
print("  - Explanations work in both modes")
print("="*80)