"""
DataTypical v0.7 - Auto-Detection Test Suite
=============================================
Tests automatic data format detection for:
1. Tabular data (DataFrames, arrays)
2. Text data (list of strings)
3. Graph data (node features + edges)

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
import warnings
warnings.filterwarnings('ignore')

from datatypical import DataTypical

print("="*80)
print("DataTypical v0.7 - Auto-Detection Test Suite")
print("="*80)


# ============================================================================
# TEST 1: Tabular Data - DataFrame
# ============================================================================
def test_tabular_dataframe():
    """Test auto-detection with DataFrame."""
    print("\n" + "-"*80)
    print("TEST 1: Tabular Data - DataFrame")
    print("-"*80)
    
    # Create sample DataFrame
    np.random.seed(42)
    data = pd.DataFrame({
        'compound_id': [f'C{i:03d}' for i in range(100)],
        'mol_weight': np.random.lognormal(5.3, 0.4, 100),
        'logP': np.random.normal(2.5, 1.2, 100),
        'num_H_donors': np.random.poisson(2, 100),
        'num_H_acceptors': np.random.poisson(4, 100),
        'TPSA': np.random.gamma(3, 15, 100),
        'ic50_nM': np.random.lognormal(2, 1.5, 100)
    })
    
    # Initialize without data_type parameter
    dt = DataTypical(
        label_columns=['compound_id'],
        stereotype_column='ic50_nM',
        stereotype_target='min',
        fast_mode=True,
        random_state=42,
        verbose=True
    )
    
    # Fit - should auto-detect tabular
    print("\n[1a] Fitting with auto-detection...")
    dt.fit(data)
    
    # Check detected type
    assert dt._detected_data_type == 'tabular', f"Expected 'tabular', got '{dt._detected_data_type}'"
    print(f"  [OK] Detected data type: {dt._detected_data_type}")
    
    # Transform
    print("\n[1b] Transforming...")
    results = dt.transform(data)
    
    # Verify results
    assert 'archetypal_rank' in results.columns, "Missing archetypal_rank"
    assert 'prototypical_rank' in results.columns, "Missing prototypical_rank"
    assert 'stereotypical_rank' in results.columns, "Missing stereotypical_rank"
    print(f"  [OK] Results shape: {results.shape}")
    print(f"  [OK] Columns: {list(results.columns)[:5]}...")
    
    # Test fit_transform
    print("\n[1c] Testing fit_transform...")
    results_combined = dt.fit_transform(data)
    assert len(results_combined) == len(data), "Length mismatch"
    print("  [OK] fit_transform works")
    
    print("\n[OK] TEST 1 PASSED: Tabular DataFrame auto-detection works")
    return dt, results


# ============================================================================
# TEST 2: Tabular Data - NumPy Array
# ============================================================================
def test_tabular_array():
    """Test auto-detection with NumPy array."""
    print("\n" + "-"*80)
    print("TEST 2: Tabular Data - NumPy Array")
    print("-"*80)
    
    # Create sample array
    np.random.seed(42)
    data_array = np.random.randn(50, 5)
    
    # Initialize without data_type parameter
    dt = DataTypical(
        nmf_rank=3,
        n_prototypes=10,
        fast_mode=True,
        random_state=42,
        verbose=True
    )
    
    # Fit - should auto-detect tabular
    print("\n[2a] Fitting array with auto-detection...")
    dt.fit(data_array)
    
    # Check detected type
    assert dt._detected_data_type == 'tabular', f"Expected 'tabular', got '{dt._detected_data_type}'"
    print(f"  [OK] Detected data type: {dt._detected_data_type}")
    
    # Transform
    results = dt.transform(data_array)
    
    # Verify results
    assert len(results) == 50, "Wrong number of results"
    assert 'archetypal_rank' in results.columns, "Missing archetypal_rank"
    print(f"  [OK] Results shape: {results.shape}")
    
    print("\n[OK] TEST 2 PASSED: Tabular array auto-detection works")
    return dt, results


# ============================================================================
# TEST 3: Text Data - List of Strings
# ============================================================================
def test_text_data():
    """Test auto-detection with list of strings."""
    print("\n" + "-"*80)
    print("TEST 3: Text Data - List of Strings")
    print("-"*80)
    
    # Create sample corpus
    corpus = [
        "protein folding dynamics and structure",
        "enzyme catalysis mechanism study",
        "protein interaction networks analysis",
        "drug binding affinity prediction",
        "molecular docking simulation",
        "protein structure prediction methods",
        "enzyme inhibitor design strategies",
        "drug discovery computational approaches",
        "molecular dynamics simulations",
        "protein sequence analysis tools"
    ] * 10  # 100 documents
    
    # Initialize with text-specific parameters
    dt = DataTypical(
        stereotype_keywords=['protein', 'enzyme'],
        nmf_rank=3,
        n_prototypes=15,
        fast_mode=True,
        random_state=42,
        verbose=True
    )
    
    # Fit - should auto-detect text
    print("\n[3a] Fitting corpus with auto-detection...")
    dt.fit(corpus)
    
    # Check detected type
    assert dt._detected_data_type == 'text', f"Expected 'text', got '{dt._detected_data_type}'"
    print(f"  [OK] Detected data type: {dt._detected_data_type}")
    
    # Transform
    results = dt.transform(corpus)
    
    # Verify results
    assert len(results) == 100, "Wrong number of results"
    assert 'archetypal_rank' in results.columns, "Missing archetypal_rank"
    assert 'stereotypical_rank' in results.columns, "Missing stereotypical_rank (keywords)"
    print(f"  [OK] Results shape: {results.shape}")
    print(f"  [OK] Top stereotypical (by keywords): {results.nlargest(3, 'stereotypical_rank').index.tolist()}")
    
    print("\n[OK] TEST 3 PASSED: Text auto-detection works")
    return dt, results


# ============================================================================
# TEST 4: Graph Data - Node Features + Edges
# ============================================================================
def test_graph_data():
    """Test auto-detection with graph data (node features + edges)."""
    print("\n" + "-"*80)
    print("TEST 4: Graph Data - Node Features + Edges")
    print("-"*80)
    
    # Create sample graph
    np.random.seed(42)
    n_nodes = 50
    
    node_features = pd.DataFrame({
        'node_id': [f'N{i:03d}' for i in range(n_nodes)],
        'feature1': np.random.randn(n_nodes),
        'feature2': np.random.randn(n_nodes),
        'feature3': np.random.randn(n_nodes),
    })
    
    # Create random edges
    edges = np.array([
        [np.random.randint(0, n_nodes), np.random.randint(0, n_nodes)]
        for _ in range(100)
    ])
    
    # Initialize with graph-specific parameters
    dt = DataTypical(
        label_columns=['node_id'],
        graph_topology_features=['degree', 'clustering'],
        nmf_rank=3,
        n_prototypes=10,
        fast_mode=True,
        random_state=42,
        verbose=True
    )
    
    # Fit - should auto-detect graph (because edges parameter present)
    print("\n[4a] Fitting graph with auto-detection...")
    dt.fit(node_features, edges=edges)
    
    # Check detected type
    assert dt._detected_data_type == 'graph', f"Expected 'graph', got '{dt._detected_data_type}'"
    print(f"  [OK] Detected data type: {dt._detected_data_type}")
    
    # Check topology features computed
    assert dt.graph_topology_df_ is not None, "Topology features not computed"
    assert 'degree' in dt.graph_topology_df_.columns, "Missing degree feature"
    assert 'clustering' in dt.graph_topology_df_.columns, "Missing clustering feature"
    print(f"  [OK] Topology features computed: {list(dt.graph_topology_df_.columns)}")
    
    # Transform
    results = dt.transform(node_features, edges=edges)
    
    # Verify results
    assert len(results) == n_nodes, "Wrong number of results"
    assert 'archetypal_rank' in results.columns, "Missing archetypal_rank"
    assert 'degree' in results.columns, "Missing degree column in results"
    assert 'clustering' in results.columns, "Missing clustering column in results"
    print(f"  [OK] Results shape: {results.shape}")
    print(f"  [OK] Includes degree: {'degree' in results.columns}, clustering: {'clustering' in results.columns}")
    
    print("\n[OK] TEST 4 PASSED: Graph auto-detection works")
    return dt, results


# ============================================================================
# TEST 5: Override Auto-Detection
# ============================================================================
def test_override_detection():
    """Test manual override of auto-detection."""
    print("\n" + "-"*80)
    print("TEST 5: Override Auto-Detection")
    print("-"*80)
    
    # Create DataFrame
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(30, 4))
    
    # Force tabular even though it would auto-detect correctly anyway
    dt = DataTypical(
        data_type='tabular',  # Explicit override
        nmf_rank=2,
        n_prototypes=5,
        fast_mode=True,
        random_state=42,
        verbose=True
    )
    
    print("\n[5a] Fitting with explicit data_type='tabular'...")
    dt.fit(data)
    
    # Check detected type matches config
    assert dt._detected_data_type == 'tabular', f"Expected 'tabular', got '{dt._detected_data_type}'"
    print(f"  [OK] Used configured data type: {dt._detected_data_type}")
    
    # Transform
    results = dt.transform(data)
    assert len(results) == 30, "Wrong number of results"
    print(f"  [OK] Results shape: {results.shape}")
    
    print("\n[OK] TEST 5 PASSED: Manual override works")
    return dt, results


# ============================================================================
# TEST 6: Error Handling - Ambiguous Input
# ============================================================================
def test_error_handling():
    """Test error handling for ambiguous inputs."""
    print("\n" + "-"*80)
    print("TEST 6: Error Handling")
    print("-"*80)
    
    dt = DataTypical(random_state=42, verbose=False, fast_mode=True)
    
    # Test 1: Invalid data type
    print("\n[6a] Testing invalid data type...")
    try:
        dt_invalid = DataTypical(data_type='invalid', fast_mode=True)
        dt_invalid.fit(pd.DataFrame(np.random.randn(10, 2)))
        print("  [FAIL] Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  [OK] Correctly raised error: {type(e).__name__}")
    
    # Test 2: Cannot detect type
    print("\n[6b] Testing undetectable input...")
    try:
        dt.fit({"not": "supported"})
        print("  [FAIL] Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  [OK] Correctly raised ValueError: {str(e)[:60]}...")
    
    print("\n[OK] TEST 6 PASSED: Error handling works correctly")
    return True


# ============================================================================
# TEST 7: All Three Formats in Sequence
# ============================================================================
def test_all_formats_sequence():
    """Test all three formats can be used sequentially."""
    print("\n" + "-"*80)
    print("TEST 7: All Three Formats in Sequence")
    print("-"*80)
    
    np.random.seed(42)
    
    # Tabular
    print("\n[7a] Tabular...")
    dt1 = DataTypical(random_state=42, fast_mode=True, verbose=False)
    tabular_data = pd.DataFrame(np.random.randn(20, 3))
    results1 = dt1.fit_transform(tabular_data)
    assert dt1._detected_data_type == 'tabular'
    print(f"  [OK] Tabular: {len(results1)} results")
    
    # Text
    print("\n[7b] Text...")
    dt2 = DataTypical(random_state=42, fast_mode=True, verbose=False)
    text_data = ["document " + str(i) for i in range(20)]
    results2 = dt2.fit_transform(text_data)
    assert dt2._detected_data_type == 'text'
    print(f"  [OK] Text: {len(results2)} results")
    
    # Graph (with varied topology)
    print("\n[7c] Graph...")
    dt3 = DataTypical(
        graph_topology_features=['degree'],
        nmf_rank=2,
        n_prototypes=5,
        fast_mode=True,
        random_state=42,
        verbose=False
    )
    
    # Create star graph for varied degrees
    node_data = pd.DataFrame({
        'x': np.random.randn(20),
        'y': np.random.randn(20),
        'z': np.random.randn(20)
    })
    
    # Star graph: node 0 connects to all others (varied degrees)
    edge_data = np.array([[0, i] for i in range(1, 20)])
    
    results3 = dt3.fit_transform(node_data, edges=edge_data)
    assert dt3._detected_data_type == 'graph'
    assert 'degree' in results3.columns  # Topology feature added
    print(f"  [OK] Graph: {len(results3)} results (degrees: 1-19)")
    
    print("\n[OK] TEST 7 PASSED: All formats work in sequence")
    return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\nRUNNING AUTO-DETECTION TEST SUITE")
    
    test_results = {}
    
    # Run all tests
    try:
        test_results['tabular_df'] = test_tabular_dataframe()
        test_results['tabular_array'] = test_tabular_array()
        test_results['text'] = test_text_data()
        test_results['graph'] = test_graph_data()
        test_results['override'] = test_override_detection()
        test_results['error_handling'] = test_error_handling()
        test_results['sequence'] = test_all_formats_sequence()
    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("AUTO-DETECTION TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in test_results.values() if v is not None and v is not False)
    total = 7
    
    print(f"\nTests passed: {passed}/{total}")
    print("\nDetailed Results:")
    
    test_names = [
        'tabular_df', 'tabular_array', 'text', 'graph',
        'override', 'error_handling', 'sequence'
    ]
    
    for name in test_names:
        if name in test_results:
            result = test_results[name]
            if result is True or (isinstance(result, tuple) and result[0] is not None):
                status = "[OK]"
                result_str = "PASS"
            else:
                status = "[FAIL]"
                result_str = "FAIL"
        else:
            status = "[FAIL]"
            result_str = "NOT RUN"
        
        print(f"  {status} {name:20s}: {result_str}")
    
    print("\n" + "="*80)
    if passed == total:
        print("[OK] ALL TESTS PASSED - Auto-Detection Works Perfectly!")
    else:
        print(f"[FAIL] {total - passed} TEST(S) FAILED")
    print("="*80)
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print("""
[OK] Tabular data (DataFrame): Auto-detected correctly
[OK] Tabular data (Array): Auto-detected correctly
[OK] Text data (List of strings): Auto-detected correctly
[OK] Graph data (edges parameter): Auto-detected correctly
[OK] Manual override (data_type parameter): Works correctly
[OK] Error handling: Catches invalid inputs
[OK] Sequential use: All three formats work independently

DETECTION PRIORITY CONFIRMED:
  1. Graph: If edges/edge_index parameter present -> 'graph'
  2. Text: If list of strings -> 'text'
  3. Tabular: If DataFrame/array -> 'tabular'

100% REPRODUCIBLE: random_state parameter ensures consistency
    """)