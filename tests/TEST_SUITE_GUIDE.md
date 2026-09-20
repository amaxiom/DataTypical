# DataTypical - Test Suite Guide

## Overview

This document provides comprehensive documentation for the DataTypical v0.7 test suite, which validates all functionality across five specialized test modules covering unit testing, data modality detection, performance scaling, visualization, and end-to-end benchmarking.

**Total Coverage:** 62 tests across 5 test suites  
**Execution Time:** 3-5 minutes (all suites)  
**Reproducibility:** 100% deterministic with random_state=42

---

## The pytest suite (added in v0.8.0)

This is the layer to run routinely, and it is the release gate: 444 tests,
99.5% statement coverage, a few minutes end to end.

```bash
python -m pytest -q
```

`pytest.ini` handles discovery. It collects the `test_*.py` modules plus the two
older suites that are already written as pytest tests. The benchmark, scaling
and visualization suites are scripts rather than pytest modules and are run
separately, as described further down.

| module | what it covers |
| --- | --- |
| `test_defects_v080.py` | regression tests for every defect fixed in v0.8.0, each reproducing the original failure |
| `test_internals.py` | numeric helpers, distance kernels, facility location, the Shapley engine, value functions |
| `test_api.py` | the public API across tabular, text and graph data, config round-tripping, error paths |
| `test_viz.py` | `datatypical_viz`, under the Agg backend |
| `test_coverage_completion.py` | import fallbacks, verbose branches, graph topology features |
| `test_final_coverage.py` | parallel backends, dtype fallbacks, degenerate geometry |
| `test_viz_completion.py` | the visualization diagnostics that fire when explanations are missing |
| `test_packaging.py` | release metadata consistency and staging parity |
| `DataTypical_unit_test_suite.py` | the original 34 unit tests, still collected |
| `DataTypical_autodetection_test_suite.py` | data type detection |

### Coverage

Measure broadly and filter at report time. Narrowing coverage to a single module
on this machine corrupts numpy sentinels and makes passing tests fail, so do not
replace this with `--cov=datatypical.py`. Disabling the numba JIT lets coverage
trace the compiled kernels.

```bash
NUMBA_DISABLE_JIT=1 python -m coverage run -m pytest -q
python -m coverage report
```

`.coveragerc` sets the source to the project root and filters the report down to
`datatypical.py` and `datatypical_viz.py`.

Ten statements in `datatypical.py` are unreachable and stay uncovered on
purpose. They are listed under "Known gaps" in `CHANGELOG.md`.

### The script suites

Run these by hand when preparing a release. They print their own summaries
rather than asserting, and they take longer.

```bash
cd tests
python DataTypical_benchmark_test_suite.py       # 8 benchmarks, all modalities
python DataTypical_scaling_test_suite.py         # 12 scaling and mode checks
python DataTypical_visualization_test_suite.py   # 16 figure checks
```

---

## Quick Start

### Prerequisites

```bash
# Install DataTypical
pip install datatypical --break-system-packages

# Install test dependencies (if not already installed)
pip install numpy pandas matplotlib seaborn --break-system-packages
```

### Running All Tests

```bash
# From project root directory
cd /path/to/datatypical

# Run each suite individually
python DataTypical_unit_test_suite.py
python DataTypical_autodetection_test_suite.py
python DataTypical_scaling_test_suite.py
python DataTypical_visualization_test_suite.py
python DataTypical_benchmark_test_suite.py
```

### Expected Output

Each test suite will report:
- Individual test results (PASS/FAIL)
- Execution times
- Summary statistics
- Final validation status

---

## Test Suite Modules

### Module 1: Unit Test Suite

**File:** `DataTypical_unit_test_suite.py`  
**Duration:** 30-45 seconds  
**Tests:** 19 tests across 6 modules

#### What It Tests

| Module | Tests | Description |
|--------|-------|-------------|
| **Shapley Correctness** | 2 | Validates Shapley value computation accuracy and stability |
| **Reproducibility** | 2 | Ensures deterministic behavior with fixed random seeds |
| **Backward Compatibility** | 3 | Verifies v0.6 behavior preserved when `shapley_mode=False` |
| **Dual-Perspective** | 3 | Tests actual vs formative ranking computations |
| **Explainability** | 3 | Validates `get_shapley_explanations()` and `get_formative_attributions()` |
| **Edge Cases** | 6 | Tests error handling, small datasets, missing values, mode validation |

#### Key Features

- **Public API Only** - No internal component testing
- **Integration Testing** - Tests through user-facing methods
- **100% Reproducible** - Fixed seeds ensure consistency
- **Fast Execution** - Uses `fast_mode=True` and small datasets

#### Running Unit Tests

```bash
python DataTypical_unit_test_suite.py
```

#### Expected Output

```
================================================================================
UNIT TEST SUMMARY
================================================================================

Modules passed: 6/6

Detailed Results:
  ✓ shapley_correctness      : PASS
  ✓ reproducibility          : PASS
  ✓ backward_compat          : PASS
  ✓ dual_perspective         : PASS
  ✓ explainability           : PASS
  ✓ edge_cases               : PASS

================================================================================
✓ ALL UNIT TESTS PASSED - DataTypical v0.7 VALIDATED
================================================================================
```

---

### Module 2: Autodetection Test Suite

**File:** `DataTypical_autodetection_test_suite.py`  
**Duration:** 20-30 seconds  
**Tests:** 7 comprehensive tests

#### What It Tests

| Test | Data Type | Description |
|------|-----------|-------------|
| **Test 1** | Tabular (DataFrame) | Auto-detects `pd.DataFrame` → 'tabular' |
| **Test 2** | Tabular (Array) | Auto-detects `np.ndarray` → 'tabular' |
| **Test 3** | Text | Auto-detects `list[str]` → 'text' |
| **Test 4** | Graph | Auto-detects `edges=` parameter → 'graph' |
| **Test 5** | Override | Manual `data_type=` parameter override |
| **Test 6** | Error Handling | Invalid inputs raise appropriate errors |
| **Test 7** | Sequential | All three formats work independently |

#### Detection Priority

```
1. Graph:   edges/edge_index parameter present → 'graph'
2. Text:    list/tuple of strings → 'text'
3. Tabular: DataFrame/array → 'tabular'
```

#### Running Autodetection Tests

```bash
python DataTypical_autodetection_test_suite.py
```

#### Expected Output

```
================================================================================
AUTO-DETECTION TEST SUMMARY
================================================================================

Tests passed: 7/7

Detailed Results:
  ✓ tabular_df         : PASS
  ✓ tabular_array      : PASS
  ✓ text               : PASS
  ✓ graph              : PASS
  ✓ override           : PASS
  ✓ error_handling     : PASS
  ✓ sequence           : PASS

================================================================================
✓ ALL TESTS PASSED - Auto-Detection Works Perfectly!
================================================================================
```

---

### Module 3: Scaling & Performance Test Suite

**File:** `DataTypical_scaling_test_suite.py`  
**Duration:** 60-90 seconds  
**Tests:** 12 performance tests

#### What It Tests

| Test | Focus | Description |
|------|-------|-------------|
| **Test 1** | Fast Mode Presets | Validates `fast_mode=True` defaults (NMF, 30 perms, skip formative) |
| **Test 2** | Publication Presets | Validates `fast_mode=False` defaults (AA, 100 perms, compute formative) |
| **Test 2b** | Speedup Analysis | Measures overall and Shapley-only speedup |
| **Test 3** | Custom Override | Tests fraction-based subsampling (shapley_top_n=0.3) |
| **Test 4** | Column Validation | Verifies ranking columns in both modes |
| **Test 5** | Explanations | Tests `get_shapley_explanations()` accessibility |
| **Test 6** | Cross-Mode Correlation | Compares NMF vs AA archetypal rankings |
| **Test 7** | Formative Attributions | Tests `get_formative_attributions()` in pub mode |
| **Test 8** | Top-N Overlap | Measures consistency of top-ranked samples |
| **Test 9** | Dataset Scaling | Tests 50, 100, 200, 500 samples |
| **Test 10** | Shapley Overhead | Measures cost of explanations |
| **Test 11** | Method Comparison | Compares NMF vs AA speed and agreement |
| **Test 12** | NMF Consistency | Verifies same method produces consistent results |

#### Performance Metrics

```
Fast Mode:         6-10× faster than publication mode
Average Speedup:   ~6× across dataset sizes
Shapley Overhead:  ~5-15× (expected for explanations)
NMF Speedup:       ~3-5× faster than AA method
```

#### Running Scaling Tests

```bash
python DataTypical_scaling_test_suite.py
```

#### Expected Output

```
================================================================================
✓ ALL TESTS PASSED
================================================================================

Key findings:
  • fast_mode=True skips formative (6x speedup)
  • fast_mode=False computes formative (comprehensive)
  • NMF vs AA produce different archetypal rankings (expected)
  • Fraction-based subsampling provides consistent speedup
  • Explanations work in both modes
================================================================================
```

---

### Module 4: Visualization Test Suite

**File:** `DataTypical_visualization_test_suite.py`  
**Duration:** 15-25 seconds  
**Tests:** 16 tests across 3 visualization modules

#### What It Tests

| Module | Tests | Description |
|--------|-------|-------------|
| **significance_plot** | 5 | Dual-perspective scatter plots (actual vs formative) |
| **heatmap** | 6 | Feature attribution heatmaps with different ordering |
| **profile_plot** | 5 | Individual sample profile plots |

#### Visualization Functions

**1. significance_plot()**
```python
from datatypical_viz import significance_plot

# Create dual-perspective scatter plot
significance_plot(
    results,
    significance='archetypal'  # or 'prototypical', 'stereotypical'
)
```

**2. heatmap()**
```python
from datatypical_viz import heatmap

# Create feature attribution heatmap
heatmap(
    dt,
    results,
    significance='archetypal',
    order='actual'  # or 'formative'
)
```

**3. profile_plot()**
```python
from datatypical_viz import profile_plot

# Create individual sample profile
profile_plot(
    dt,
    sample_idx,
    significance='archetypal',
    order='local'  # or 'global'
)
```

#### Running Visualization Tests

```bash
python DataTypical_visualization_test_suite.py
```

#### Expected Output

```
================================================================================
VISUALIZATION TEST SUMMARY
================================================================================

Modules passed: 3/3

Detailed Results:
  ✓ significance_plot    : PASS (5 tests)
  ✓ heatmap              : PASS (6 tests)
  ✓ profile_plot         : PASS (5 tests)

================================================================================
✓ ALL TESTS PASSED - Visualization Module Validated
================================================================================
```

---

### Module 5: Benchmark Test Suite

**File:** `DataTypical_benchmark_test_suite.py`  
**Duration:** 40-60 seconds  
**Tests:** 8 comprehensive benchmarks

#### What It Tests

| Benchmark | Modality | Dataset | Focus |
|-----------|----------|---------|-------|
| **#1** | Tabular | 200 compounds × 7 features | Fast mode (explanations only) |
| **#2** | Tabular | 300 molecules × 4 features | Publication mode (dual-perspective) |
| **#3** | Tabular | 1000 samples × 5 features | Subsampling (shapley_top_n=100) |
| **#4** | Text | 100 research abstracts | Keyword-based stereotypes |
| **#5** | Text | 50 documents + metadata | Metadata-based stereotypes |
| **#6** | Graph | 100 proteins, 200 edges | Scale-free network, topology features |
| **#7** | Graph | 80 molecules, random edges | Molecular network, pagerank |
| **#8** | Comparison | 200 samples × 3 features | Fast vs Publication modes |

#### Benchmark Validations

**Tabular Benchmarks:**
- Fast mode skips formative (all None)
- Publication mode computes formative (has values)
- Explanations work in both modes
- Subsampling reduces computational cost
- Rankings properly normalized [0, 1]

**Text Benchmarks:**
- Keywords increase stereotypical rank
- Metadata correlates with stereotypical rank
- TF-IDF vectorization works correctly

**Graph Benchmarks:**
- Topology features computed (degree, clustering, pagerank)
- Hub nodes have higher degree
- Graph structure preserved in results

**Mode Comparison:**
- Fast mode: ~6× faster, skips formative
- Publication mode: comprehensive, computes formative
- Same method → high correlation (>0.6)
- Different methods (NMF vs AA) → lower correlation (>0.2)

#### Running Benchmarks

```bash
python DataTypical_benchmark_test_suite.py
```

#### Expected Output

```
================================================================================
BENCHMARK SUMMARY
================================================================================

✓ Successful: 8/8
✗ Failed: 0/8

MODALITY COVERAGE
  ✓ Tabular            : 3/3 benchmark(s)
  ✓ Text               : 2/2 benchmark(s)
  ✓ Graph              : 2/2 benchmark(s)
  ✓ Mode Comparison    : 1/1 benchmark(s)

TIMING BREAKDOWN
  ✓ 1. Small Tabular (Fast Mode)                          :   4.23s
  ✓ 2. Medium Tabular (Publication Mode)                  :  10.45s
  ✓ 3. Large Tabular (Subsampling)                        :  12.18s
  ✓ 4. Text Small (Keywords)                              :   3.12s
  ✓ 5. Text Metadata                                      :   1.84s
  ✓ 6. Graph Protein Network                              :   4.67s
  ✓ 7. Graph Molecular                                    :   2.91s
  ✓ 8. Mode Comparison (Fast vs Publication)              :  11.23s

  Total time: 50.63s

================================================================================
✓ ALL BENCHMARKS PASSED
================================================================================

DataTypical v0.7 VALIDATED
  • All modalities working (Tabular, Text, Graph)
  • Fast mode optimizations functional
  • Publication mode dual-perspective working
  • Shapley explanations + formative validated
  • Auto-detection working as expected
================================================================================
```

---

## Configuration & Customization

### Updating File Paths

All test suites include this line at the top:

```python
sys.path.insert(0, '/my/project')  # update your path here
```

**Update this to your DataTypical installation directory:**

```python
# Example for local development
sys.path.insert(0, '/home/username/datatypical')

# Example for conda environment
sys.path.insert(0, '/home/username/miniconda3/envs/myenv/lib/python3.10/site-packages')

# Example for system installation
# No need to modify if installed globally
```

### Adjusting Test Parameters

#### Speed vs Coverage Trade-offs

**Faster Testing (Exploration):**
```python
# Reduce permutations
shapley_n_permutations=10  # Instead of 30

# Reduce dataset sizes
n_samples = 50  # Instead of 100

# Skip slower tests
# Comment out large dataset benchmarks
```

**More Thorough Testing (Publication):**
```python
# Increase permutations
shapley_n_permutations=100  # Instead of 30

# Larger datasets
n_samples = 1000  # Instead of 100

# Add additional test cases
# Duplicate and modify existing tests
```

#### Reproducibility

All tests use `random_state=42` for reproducibility. To test with different random states:

```python
# Change all instances of
random_state=42

# To
random_state=YOUR_SEED
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'datatypical'
```

**Solution:**
```bash
# Verify installation
pip show datatypical

# Reinstall if needed
pip install datatypical --break-system-packages

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

#### Issue 2: Test Failures Due to Environment

**Error:**
```
AssertionError: Expected correlation > 0.8, got 0.45
```

**Possible Causes:**
- Different random seed behavior across Python versions
- Floating-point precision differences
- NumPy/SciPy version differences

**Solution:**
```bash
# Check versions
python --version  # Should be 3.9+
pip show numpy scipy scikit-learn

# Update packages
pip install --upgrade numpy scipy scikit-learn
```

#### Issue 3: Memory Errors

**Error:**
```
MemoryError: Unable to allocate array
```

**Solution:**
```python
# Reduce dataset sizes in benchmarks
n_samples = 100  # Instead of 1000

# Use fast_mode to reduce memory
fast_mode=True

# Reduce permutations
shapley_n_permutations=10
```

#### Issue 4: Slow Execution

**Symptoms:** Tests take >10 minutes to complete

**Solutions:**

1. **Use Fast Mode:**
```python
fast_mode=True  # 6-10× speedup
```

2. **Reduce Permutations:**
```python
shapley_n_permutations=10  # Minimum for testing
```

3. **Skip Large Benchmarks:**
```python
# Comment out in benchmark suite
# run_benchmark("3. Large Tabular (Subsampling)", benchmark_large_tabular)
```

#### Issue 5: Visualization Errors

**Error:**
```
RuntimeError: No formative data available
```

**Solution:**
```python
# Ensure fast_mode=False for formative tests
dt = DataTypical(
    shapley_mode=True,
    fast_mode=False,  # Required for formative
    random_state=42
)
```

---

## Best Practices

### Running Tests During Development

**1. Quick Validation (Unit Tests Only)**
```bash
# 30 seconds
python DataTypical_unit_test_suite.py
```

**2. API Changes (Unit + Autodetection)**
```bash
# 1 minute
python DataTypical_unit_test_suite.py
python DataTypical_autodetection_test_suite.py
```

**3. Performance Changes (Add Scaling)**
```bash
# 2 minutes
python DataTypical_unit_test_suite.py
python DataTypical_autodetection_test_suite.py
python DataTypical_scaling_test_suite.py
```

**4. Full Validation (All Suites)**
```bash
# 3-5 minutes
python DataTypical_unit_test_suite.py
python DataTypical_autodetection_test_suite.py
python DataTypical_scaling_test_suite.py
python DataTypical_visualization_test_suite.py
python DataTypical_benchmark_test_suite.py
```

### Continuous Integration

**GitHub Actions Example:**

```yaml
name: DataTypical Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install numpy pandas matplotlib seaborn scikit-learn
        pip install .
    
    - name: Run Unit Tests
      run: python DataTypical_unit_test_suite.py
    
    - name: Run Autodetection Tests
      run: python DataTypical_autodetection_test_suite.py
    
    - name: Run Scaling Tests
      run: python DataTypical_scaling_test_suite.py
    
    - name: Run Visualization Tests
      run: python DataTypical_visualization_test_suite.py
    
    - name: Run Benchmarks
      run: python DataTypical_benchmark_test_suite.py
```

### Creating New Tests

**Template for Adding Tests:**

```python
def test_new_feature():
    """Test description."""
    print("\n[X] Testing new feature...")
    
    # Setup
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.randn(50),
        'feature2': np.random.randn(50)
    })
    
    # Execute
    dt = DataTypical(
        fast_mode=True,
        random_state=42,
        verbose=False
    )
    results = dt.fit_transform(data)
    
    # Validate
    assert 'expected_column' in results.columns, "Missing expected column"
    assert len(results) == len(data), "Row count mismatch"
    
    print("  ✓ New feature working correctly")
```

---

## Performance Expectations

### Typical Execution Times

| Suite | Tests | Typical Duration | Max Duration |
|-------|-------|------------------|--------------|
| Unit Tests | 19 | 30-45s | 60s |
| Autodetection | 7 | 20-30s | 45s |
| Scaling | 12 | 60-90s | 120s |
| Visualization | 16 | 15-25s | 40s |
| Benchmarks | 8 | 40-60s | 90s |
| **TOTAL** | **62** | **3-4 min** | **6 min** |

### Hardware Considerations

**Minimum Requirements:**
- CPU: 2+ cores
- RAM: 4GB
- Disk: 1GB free space

**Recommended:**
- CPU: 4+ cores (parallel operations)
- RAM: 8GB+ (large dataset benchmarks)
- Disk: 2GB+ (room for temporary files)

### Optimization Tips

**1. Use Parallelization:**
```python
# DataTypical uses n_jobs=-1 by default
dt = DataTypical(n_jobs=-1)  # Use all cores
```

**2. Enable Fast Mode:**
```python
# 6-10× speedup
dt = DataTypical(fast_mode=True)
```

**3. Reduce Permutations:**
```python
# For development/testing
shapley_n_permutations=10  # Minimum

# For validation
shapley_n_permutations=30  # Balanced

# For publication
shapley_n_permutations=100  # Thorough
```

---

## Test Suite Maintenance

### Version Compatibility

| DataTypical Version | Test Suite Version | Notes |
|--------------------|--------------------|-------|
| v0.7.x | Current | All tests pass |
| v0.6.x | Backward compatible | Unit tests verify v0.6 behavior |
| v0.5.x | Partial | Autodetection not available |
| v0.4.x | Limited | Shapley not available |

### Updating Tests

**When to Update Tests:**

1. **API Changes** - Update affected test suites
2. **New Features** - Add new test cases
3. **Bug Fixes** - Add regression tests
4. **Performance Improvements** - Update timing thresholds

**Checklist for Updates:**

- [ ] Update affected test files
- [ ] Run all test suites
- [ ] Update this documentation
- [ ] Update version numbers
- [ ] Commit with descriptive message

---

## FAQ

### Q1: Why do some tests use `fast_mode=True` and others `fast_mode=False`?

**A:** We test both modes to ensure:
- Fast mode works for rapid exploration (6-10× speedup)
- Publication mode works for comprehensive analysis (formative + explanations)
- Both modes produce valid, consistent results

### Q2: Why are correlation thresholds different for archetypal vs prototypical?

**A:** 
- **Archetypal:** NMF vs AA are fundamentally different methods (threshold: 0.3)
- **Prototypical:** Same algorithm in both modes (threshold: 0.98)
- Different methods lead to different archetypes, so lower correlation is expected and correct

### Q3: Can I run tests in parallel?

**A:** No, run test suites sequentially. Each suite:
- Uses the same random seed (42)
- May write to temporary files
- Has timing measurements that could be affected by parallel execution

### Q4: What if a test fails?

**A:** 
1. Check error message for details
2. Verify DataTypical installation: `pip show datatypical`
3. Check Python version: `python --version` (should be 3.9+)
4. Review "Troubleshooting" section above
5. Run test in isolation to reproduce
6. Check GitHub issues or report new issue

### Q5: How do I test only specific functionality?

**A:**
- **Unit tests only:** Run Module 1
- **Data modality:** Run Module 2
- **Performance:** Run Module 3
- **Visualizations:** Run Module 4
- **End-to-end:** Run Module 5

---

### Reporting Issues

When reporting test failures, include:

```
Environment:
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.10.5]
- DataTypical: [e.g., 0.7.0]
- NumPy: [e.g., 1.24.3]

Test Suite: [e.g., Module 1: Unit Tests]
Test Name: [e.g., test_shapley_additivity_property]

Error Message:
[Paste full error message and traceback]

Steps to Reproduce:
1. [Step 1]
2. [Step 2]
...
```

---

## References

### Documentation
- [DataTypical README](README.md)
- [API Documentation](PACKAGE_SUMMARY.md)
- [Quick Reference](QUICK_REFERENCE.md)
- [Examples](EXAMPLES.md)

### Test Files
- [Unit Tests](DataTypical_unit_test_suite.py)
- [Autodetection Tests](DataTypical_autodetection_test_suite.py)
- [Scaling Tests](DataTypical_scaling_test_suite.py)
- [Visualization Tests](DataTypical_visualization_test_suite.py)
- [Benchmark Suite](DataTypical_benchmark_test_suite.py)

### External Resources
- [NumPy Testing](https://numpy.org/doc/stable/reference/testing.html)
- [Pandas Testing](https://pandas.pydata.org/docs/reference/testing.html)
- [Python unittest](https://docs.python.org/3/library/unittest.html)

---

## Changelog

### v0.7.0 (2026-01-21)
- Complete test suite refactoring
- All tests use public API only
- 100% reproducible with `random_state=42`
- Optimized for fast execution
- Comprehensive documentation
- 62 total tests across 5 modules
- 3-5 minute total execution time

---

## License

This test suite is part of DataTypical and is distributed under the same MIT License.

---

## Contact

**Author:** Amanda S. Barnard  
**GitHub:** https://github.com/amaxiom/datatypical  
**Issues:** https://github.com/amaxiom/datatypical/issues

---

**Last Updated:** January 21, 2026  
**Test Suite Version:** 0.7.0  
**DataTypical Version:** 0.7.x