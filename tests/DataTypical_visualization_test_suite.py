"""
DataTypical v0.7 - Visualization Test Suite
==========================================
Complete test coverage for all visualization functions.
Run this after unit tests and benchmarks to validate visualization layer.

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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datatypical import DataTypical
from datatypical_viz import (
    significance_plot,
    heatmap, 
    profile_plot,
)

print("="*80)
print("DataTypical v0.7 - Visualization Test Suite")
print("="*80)

# ============================================================================
# Setup Test Data
# ============================================================================
print("\nRUNNING VISUALIZATION TEST SUITE")

np.random.seed(42)
n_samples = 100  # Optimized for fast testing
n_features = 6

test_data = pd.DataFrame({
    'id': [f'S{i:03d}' for i in range(n_samples)],
    'feat_A': np.random.randn(n_samples),
    'feat_B': np.random.randn(n_samples),
    'feat_C': np.random.randn(n_samples),
    'feat_D': np.random.randn(n_samples),
    'feat_E': np.random.randn(n_samples),
    'feat_F': np.random.randn(n_samples),
    'property': np.random.randn(n_samples),
    'target': np.random.randn(n_samples)
})

# Fit models
dt_no_shapley = DataTypical(
    label_columns=['id'],
    shapley_mode=False,
    fast_mode=True,
    random_state=42,
    verbose=False
)
results_no_shapley = dt_no_shapley.fit_transform(test_data)

dt_shapley = DataTypical(
    label_columns=['id'],
    stereotype_column='target',
    stereotype_target='max',
    shapley_mode=True,
    shapley_n_permutations=10,  # Reduced for fast testing
    fast_mode=False,
    n_jobs=1,  # Single job to avoid timeout
    random_state=42,
    verbose=False
)
results_shapley = dt_shapley.fit_transform(test_data)

print(f"[OK] Data prepared: {n_samples} samples, {n_features} features")

# ============================================================================
# MODULE 1: significance_plot
# ============================================================================
print("\n" + "-"*80)
print("TEST 1: significance_plot Tests")
print("-"*80)

mod1_tests = 0
mod1_passed = 0

# Test 1.1: Basic plot
print("\n[1.1] Basic significance plot")
try:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax = significance_plot(
        results_shapley,
        significance='archetypal',
        ax=ax
    )
    assert ax is not None
    plt.close(fig)
    print("  [OK] Basic plot created")
    mod1_passed += 1
except Exception as e:
    print(f"  FAILED: {e}")
mod1_tests += 1

# Test 1.2: Color by property
print("\n[1.2] Color by property")
try:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax = significance_plot(
        results_shapley,
        significance='prototypical',
        color_by='property',
        ax=ax
    )
    assert ax is not None
    plt.close(fig)
    print("  [OK] Color-coded plot created")
    mod1_passed += 1
except Exception as e:
    print(f"  FAILED: {e}")
mod1_tests += 1

# Test 1.3: Quadrant lines
print("\n[1.3] Quadrant lines")
try:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax = significance_plot(
        results_shapley,
        significance='archetypal',
        quadrant_lines=True,
        quadrant_threshold=(0.5, 0.5),
        ax=ax
    )
    lines = ax.get_lines()
    assert len(lines) >= 2, "Expected quadrant lines"
    plt.close(fig)
    print("  [OK] Quadrant lines added")
    mod1_passed += 1
except Exception as e:
    print(f"  FAILED: {e}")
mod1_tests += 1

# Test 1.4: Invalid significance handling
print("\n[1.4] Invalid significance error handling")
try:
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        significance_plot(
            results_shapley,
            significance='nonexistent',
            ax=ax
        )
        print("  FAILED: Should raise ValueError")
    except ValueError:
        print("  [OK] Correctly raised ValueError")
        mod1_passed += 1
    plt.close(fig)
except Exception as e:
    print(f"  FAILED: {e}")
mod1_tests += 1

# Test 1.5: All three significance types
print("\n[1.5] All three significance types")
try:
    types_ok = 0
    for sig_type in ['archetypal', 'prototypical', 'stereotypical']:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax = significance_plot(
            results_shapley,
            significance=sig_type,
            ax=ax
        )
        assert ax is not None
        plt.close(fig)
        types_ok += 1
    
    assert types_ok == 3
    print("  [OK] All three types work")
    mod1_passed += 1
except Exception as e:
    print(f"  FAILED: {e}")
mod1_tests += 1

print(f"\n-> Module 1: {mod1_passed}/{mod1_tests} tests passed")

# ============================================================================
# MODULE 2: heatmap
# ============================================================================
print("\n" + "-"*80)
print("TEST 2: heatmap Tests")
print("-"*80)

mod2_tests = 0
mod2_passed = 0

# Test 2.1: Archetypal with actual ordering
print("\n[2.1] Archetypal heatmap (actual ordering)")
try:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = heatmap(
        dt_shapley,
        results_shapley,
        significance='archetypal',
        order='actual',
        top_n=10,
        ax=ax
    )
    assert ax is not None
    title = ax.get_title()
    assert 'archetypal' in title.lower()
    plt.close(fig)
    print("  [OK] Archetypal heatmap created")
    mod2_passed += 1
except Exception as e:
    print(f"  FAILED: {e}")
mod2_tests += 1

# Test 2.2: Prototypical with formative ordering
print("\n[2.2] Prototypical heatmap (formative ordering)")
try:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = heatmap(
        dt_shapley,
        results_shapley,
        significance='prototypical',
        order='formative',
        top_n=8,
        ax=ax
    )
    assert ax is not None
    plt.close(fig)
    print("  [OK] Prototypical formative heatmap created")
    mod2_passed += 1
except Exception as e:
    print(f"  FAILED: {e}")
mod2_tests += 1

# Test 2.3: Stereotypical heatmap
print("\n[2.3] Stereotypical heatmap")
try:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = heatmap(
        dt_shapley,
        results_shapley,
        significance='stereotypical',
        order='actual',
        top_n=5,
        ax=ax
    )
    assert ax is not None
    plt.close(fig)
    print("  [OK] Stereotypical heatmap created")
    mod2_passed += 1
except Exception as e:
    print(f"  FAILED: {e}")
mod2_tests += 1

# Test 2.4: Custom sample selection
print("\n[2.4] Custom sample selection")
try:
    fig, ax = plt.subplots(figsize=(8, 6))
    custom_samples = [0, 5, 10, 15, 20]
    ax = heatmap(
        dt_shapley,
        results_shapley,
        samples=custom_samples,
        significance='archetypal',
        order='actual',
        ax=ax
    )
    assert ax is not None
    plt.close(fig)
    print("  [OK] Custom sample selection works")
    mod2_passed += 1
except Exception as e:
    print(f"  FAILED: {e}")
mod2_tests += 1

# Test 2.5: Top features filtering
print("\n[2.5] Top features filtering")
try:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = heatmap(
        dt_shapley,
        results_shapley,
        significance='archetypal',
        order='actual',
        top_n=10,
        top_features=5,
        ax=ax
    )
    assert ax is not None
    plt.close(fig)
    print("  [OK] Top features filtering works")
    mod2_passed += 1
except Exception as e:
    print(f"  FAILED: {e}")
mod2_tests += 1

# Test 2.6: Error without Shapley
print("\n[2.6] Error without Shapley mode")
try:
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        heatmap(
            dt_no_shapley,
            results_no_shapley,
            significance='archetypal',
            order='actual',
            top_n=10,
            ax=ax
        )
        print("  FAILED: Should raise RuntimeError")
    except RuntimeError:
        print("  [OK] Correctly raised RuntimeError")
        mod2_passed += 1
    plt.close(fig)
except Exception as e:
    print(f"  FAILED: {e}")
mod2_tests += 1

print(f"\n-> Module 2: {mod2_passed}/{mod2_tests} tests passed")

# ============================================================================
# MODULE 3: profile_plot
# ============================================================================
print("\n" + "-"*80)
print("TEST 3: profile_plot Tests")
print("-"*80)

mod3_tests = 0
mod3_passed = 0

# Test 3.1: Archetypal profile with local ordering
print("\n[3.1] Archetypal profile (local ordering)")
try:
    fig, ax = plt.subplots(figsize=(12, 5))
    top_idx = results_shapley['archetypal_rank'].idxmax()
    ax = profile_plot(
        dt_shapley,
        sample_idx=top_idx,
        significance='archetypal',
        order='local',
        ax=ax
    )
    assert ax is not None
    assert 'feature' in ax.get_xlabel().lower()
    plt.close(fig)
    print("  [OK] Archetypal profile created")
    mod3_passed += 1
except Exception as e:
    print(f"  FAILED: {e}")
mod3_tests += 1

# Test 3.2: Prototypical profile with global ordering
print("\n[3.2] Prototypical profile (global ordering)")
try:
    fig, ax = plt.subplots(figsize=(12, 5))
    top_idx = results_shapley['prototypical_rank'].idxmax()
    ax = profile_plot(
        dt_shapley,
        sample_idx=top_idx,
        significance='prototypical',
        order='global',
        ax=ax
    )
    assert ax is not None
    title = ax.get_title()
    assert 'prototypical' in title.lower()
    plt.close(fig)
    print("  [OK] Prototypical profile created")
    mod3_passed += 1
except Exception as e:
    print(f"  FAILED: {e}")
mod3_tests += 1

# Test 3.3: Stereotypical profile
print("\n[3.3] Stereotypical profile")
try:
    fig, ax = plt.subplots(figsize=(12, 5))
    top_idx = results_shapley['stereotypical_rank'].idxmax()
    ax = profile_plot(
        dt_shapley,
        sample_idx=top_idx,
        significance='stereotypical',
        order='local',
        ax=ax
    )
    assert ax is not None
    plt.close(fig)
    print("  [OK] Stereotypical profile created")
    mod3_passed += 1
except Exception as e:
    print(f"  FAILED: {e}")
mod3_tests += 1

# Test 3.4: Zero reference line
print("\n[3.4] Zero reference line present")
try:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax = profile_plot(
        dt_shapley,
        sample_idx=0,
        significance='archetypal',
        order='local',
        ax=ax
    )
    # Check for horizontal line at y=0
    lines = ax.get_lines()
    has_zero_line = any(
        abs(line.get_ydata()[0]) < 0.01 and abs(line.get_ydata()[-1]) < 0.01
        for line in lines
    )
    assert has_zero_line, "Zero reference line not found"
    plt.close(fig)
    print("  [OK] Zero reference line present")
    mod3_passed += 1
except Exception as e:
    print(f"  FAILED: {e}")
mod3_tests += 1

# Test 3.5: Error without Shapley
print("\n[3.5] Error without Shapley mode")
try:
    fig, ax = plt.subplots(figsize=(12, 5))
    try:
        profile_plot(
            dt_no_shapley,
            sample_idx=0,
            significance='archetypal',
            order='local',
            ax=ax
        )
        print("  FAILED: Should raise RuntimeError")
    except RuntimeError:
        print("  [OK] Correctly raised RuntimeError")
        mod3_passed += 1
    plt.close(fig)
except Exception as e:
    print(f"  FAILED: {e}")
mod3_tests += 1

print(f"\n-> Module 3: {mod3_passed}/{mod3_tests} tests passed")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("VISUALIZATION TEST SUITE SUMMARY")
print("="*80)

modules = [
    ("Module 1: significance_plot", mod1_tests, mod1_passed),
    ("Module 2: heatmap", mod2_tests, mod2_passed),
    ("Module 3: profile_plot", mod3_tests, mod3_passed),
]

total_tests = sum(m[1] for m in modules)
total_passed = sum(m[2] for m in modules)

print(f"\nModules tested: 3/3")
for name, tests, passed in modules:
    status = "[OK]" if tests == passed else "[FAIL]"
    print(f"  {status} {name}: {passed}/{tests} tests passed")

print(f"\nOverall: {total_passed}/{total_tests} tests passed")

if total_passed == total_tests:
    print("\n" + "="*80)
    print("[OK] ALL VISUALIZATION TESTS PASSED")
    print("="*80)
    print("\nVisualization functions validated:")
    print("  [OK] significance_plot - Dual-perspective scatter")
    print("  [OK] heatmap - Feature attribution heatmap")
    print("  [OK] profile_plot - Individual sample profile")
else:
    print("\n" + "="*80)
    print(f"[FAIL] SOME TESTS FAILED: {total_tests - total_passed} failures")
    print("="*80)

print("\n" + "="*80)