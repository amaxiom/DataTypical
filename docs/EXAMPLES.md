# DataTypical Examples

**Complete worked examples demonstrating discovery workflows across scientific domains**

---

## Table of Contents

1. [Drug Discovery: Alternative Mechanisms](#example-1-drug-discovery-alternative-mechanisms)
2. [Materials Science: Boundary Extremes](#example-2-materials-science-boundary-extremes)
3. [Quality Control: Novel vs Known Variation](#example-3-quality-control-novel-vs-known-variation)
4. [Biomarker Discovery: Pattern Exploration](#example-4-biomarker-discovery-pattern-exploration)
5. [Dataset Curation: Coverage Gaps](#example-5-dataset-curation-coverage-gaps)

---

## Example 1: Drug Discovery: Alternative Mechanisms

**Goal**: Find compounds with high activity and understand if they achieve activity through different mechanisms.

### The Dataset
```python
import pandas as pd
import numpy as np
from datatypical import DataTypical
from datatypical_viz import significance_plot, heatmap, profile_plot
import matplotlib.pyplot as plt

# Load compound library
compounds = pd.read_csv('compound_library.csv')
print(f"Dataset: {compounds.shape[0]} compounds × {compounds.shape[1]} features")

# Features: molecular descriptors (logP, MW, HBA, HBD, TPSA, etc.)
# Target: IC50 activity values
print(compounds.columns)
# ['compound_id', 'logP', 'MW', 'HBA', 'HBD', 'TPSA', 'rotatable_bonds', 
#  'aromatic_rings', 'IC50', 'activity']
```

### Step 1: Exploration (Fast)
```python
# Quick analysis to identify active compounds
dt = DataTypical(
    stereotype_column='activity',  # Binary: active (1) or inactive (0)
    fast_mode=True
)

# Separate features from labels
feature_cols = ['logP', 'MW', 'HBA', 'HBD', 'TPSA', 'rotatable_bonds', 'aromatic_rings']
X = compounds[feature_cols]
results = dt.fit_transform(X)

# Merge with original data
compounds_results = pd.concat([compounds, results], axis=1)

# Top stereotypical (activity-like) compounds
top_active = compounds_results.nlargest(20, 'stereotypical_rank')
print(f"\nTop 20 activity-like compounds:")
print(top_active[['compound_id', 'activity', 'stereotypical_rank', 'IC50']])
```

**Output**:
```
Top 20 activity-like compounds:
    compound_id  activity  stereotypical_rank     IC50
142  CPD_0142         1              0.987     0.12 nM
289  CPD_0289         1              0.981     0.08 nM
034  CPD_0034         1              0.976     0.15 nM
...
```

### Step 2: Understanding Mechanisms (With Shapley)
```python
# Detailed analysis with explanations
dt = DataTypical(
    shapley_mode=True,
    stereotype_column='activity',
    fast_mode=False
)
results = dt.fit_transform(X)
compounds_results = pd.concat([compounds, results], axis=1)

# Visualize: Are high-activity compounds redundant or diverse?
fig, ax = plt.subplots(figsize=(8, 6))
significance_plot(
    results, 
    significance='stereotypical',
    color_by='IC50',
    label_top=5,
    ax=ax
)
ax.figure.savefig('stereotypical_scatter.png', dpi=300, bbox_inches='tight')
```

**What We See**:
- **Top-right quadrant**: 8 critical compounds (high actual + high formative)
  - These are unique mechanisms - irreplaceable
- **Bottom-right quadrant**: 15 replaceable compounds (high actual + low formative)
  - Active, but similar to each other
- **Top-left quadrant**: 3 gap fillers (low actual + high formative)
  - Moderate activity but structurally important

### Step 3: Feature Patterns Across Active Compounds
```python
# Heatmap of top stereotypical compounds
fig, ax = plt.subplots(figsize=(12, 8))
heatmap(
    dt, results,
    significance='stereotypical',
    order='actual',
    top_n=25,
    ax=ax
)
ax.figure.savefig('stereotypical_heatmap.png', dpi=300, bbox_inches='tight')
```

**What We See**:
```
Feature Importance (left to right):
1. logP (most important)
2. aromatic_rings
3. HBA
4. TPSA
5. MW
...

Pattern Recognition:
- Vertical clusters: 3 distinct groups
  Group A (samples 0-8): High logP, low TPSA
  Group B (samples 9-17): Moderate logP, high aromatic_rings
  Group C (samples 18-25): Low MW, high HBA
```

**Discovery**: Three different structural classes achieve high activity!

### Step 4: Profiling Alternative Mechanisms
```python
# Find high-formative stereotypical compounds (unique mechanisms)
unique_mechanisms = compounds_results[
    (compounds_results['stereotypical_shapley_rank'] > 0.8)
].nlargest(3, 'stereotypical_rank')

print("\nUnique mechanism compounds:")
print(unique_mechanisms[['compound_id', 'stereotypical_rank', 'stereotypical_shapley_rank']])

# Profile each mechanism
fig, axes = plt.subplots(3, 1, figsize=(12, 15))
for i, idx in enumerate(unique_mechanisms.index):
    profile_plot(
        dt, idx,
        significance='stereotypical',
        order='local',
        ax=axes[i]
    )
    compound_id = compounds.loc[idx, 'compound_id']
    axes[i].set_title(f'Mechanism {i+1}: {compound_id}', fontsize=16)

plt.tight_layout()
fig.savefig('alternative_mechanisms.png', dpi=300, bbox_inches='tight')
```

**What We See**:

**Mechanism 1** (CPD_0142):
- Dominated by positive logP contribution (high lipophilicity)
- Low TPSA (good membrane permeability)
- Small MW
- **Interpretation**: Lipophilic small molecule pathway

**Mechanism 2** (CPD_0289):
- High aromatic_rings contribution
- Moderate logP
- High HBA (hydrogen bond acceptors)
- **Interpretation**: Aromatic pi-stacking pathway

**Mechanism 3** (CPD_0034):
- High HBD and HBA contributions (polar interactions)
- Low logP (hydrophilic)
- Multiple rotatable_bonds (flexibility)
- **Interpretation**: Polar flexible binding pathway

### Discovery Summary

**What We Found**:
1. **Three distinct mechanisms** for achieving high activity
2. **8 irreplaceable compounds** defining these mechanisms
3. **15 redundant compounds** that cluster with known mechanisms
4. **Actionable insight**: Focus synthesis on exploring the three mechanism spaces

**Dataset Curation Decision**:
- Keep all 8 critical compounds (one can't replace another)
- Keep 1 representative per cluster from the 15 redundant
- Final curated set: 8 + 5 = 13 compounds instead of 23

**Time Investment**: 15 minutes → Saved weeks of redundant synthesis

---

## Example 2: Materials Science: Boundary Extremes

**Goal**: Identify materials with extreme property combinations for targeted applications.

### The Dataset
```python
# Thermoelectric materials database
materials = pd.read_csv('thermoelectric_materials.csv')
print(f"Dataset: {materials.shape[0]} materials × {materials.shape[1]} properties")

# Properties: electrical conductivity, thermal conductivity, Seebeck coefficient, etc.
print(materials.columns)
# ['material_id', 'composition', 'sigma', 'kappa', 'S', 'PF', 'ZT', 'band_gap', 'carrier_conc']
```

### Step 1: Find Boundary Extremes
```python
# Archetypal analysis for extreme property combinations
dt = DataTypical(
    shapley_mode=True,
    fast_mode=False,
    nmf_rank=10  # More corners (archetypes) for complex property space
)

feature_cols = ['sigma', 'kappa', 'S', 'band_gap', 'carrier_conc']
X = materials[feature_cols]
results = dt.fit_transform(X)
materials_results = pd.concat([materials, results], axis=1)

# Find most extreme materials
extreme_materials = materials_results.nlargest(15, 'archetypal_rank')
print("\nMost extreme materials:")
print(extreme_materials[['material_id', 'composition', 'archetypal_rank', 'ZT']])
```

**Output**:
```
Most extreme materials:
    material_id        composition  archetypal_rank    ZT
45   MAT_045      Bi2Te3-doped          0.973      2.1
89   MAT_089      SnSe-pristine         0.968      2.4
123  MAT_123      PbTe-Na               0.961      1.9
...
```

### Step 2: Dual Perspective Analysis
```python
# Visualize: Which extremes are unique vs clustered?
fig, ax = plt.subplots(figsize=(8, 6))
significance_plot(
    results,
    significance='archetypal',
    color_by='ZT',
    label_top=10,
    ax=ax
)
ax.figure.savefig('archetypal_scatter.png', dpi=300, bbox_inches='tight')
```

**What We See**:
- **Critical materials** (top-right): 5 materials at unique boundary positions
  - Each defines a different "corner" of property space
  - Removing any would shrink the known space
- **Replaceable extremes** (bottom-right): 10 materials clustered in 3 groups
  - All high-performing but similar to others
  - Keep one per cluster

### Step 3: Understanding Property Trade-offs
```python
# Profile a critical boundary material
critical_idx = materials_results[
    (materials_results['archetypal_rank'] > 0.9) &
    (materials_results['archetypal_shapley_rank'] > 0.9)
].index[0]

print(f"\nProfiling material: {materials.loc[critical_idx, 'material_id']}")
print(f"Composition: {materials.loc[critical_idx, 'composition']}")
print(f"ZT: {materials.loc[critical_idx, 'ZT']}")

fig, ax = plt.subplots(figsize=(12, 5))
profile_plot(dt, critical_idx, significance='archetypal', order='local', ax=ax)
ax.figure.savefig('critical_material_profile.png', dpi=300, bbox_inches='tight')
```

**What We See**:
```
Feature Contributions (ordered by local importance):
1. carrier_conc (huge positive) → Extremely high carrier concentration
2. sigma (large positive) → Very high electrical conductivity
3. kappa (large negative) → Extremely low thermal conductivity
4. S (moderate positive) → High Seebeck coefficient
5. band_gap (small negative) → Narrow band gap

Interpretation:
This material achieves extreme performance through:
- Unprecedented carrier concentration (10× typical)
- Unusual decoupling of electrical and thermal conductivity
- Optimal band gap engineering
```

### Step 4: Identify Gaps in Property Space
```python
# Find gap-filling materials (low archetypal, high formative)
gap_fillers = materials_results[
    (materials_results['archetypal_rank'] < 0.3) &
    (materials_results['archetypal_shapley_rank'] > 0.7)
]

print(f"\nFound {len(gap_fillers)} gap-filling materials")
print(gap_fillers[['material_id', 'composition', 'archetypal_rank', 'archetypal_shapley_rank']])

# Heatmap of gap fillers
fig, ax = plt.subplots(figsize=(12, 6))
heatmap(
    dt, results,
    significance='archetypal',
    order='formative',  # Order by structure-creation
    top_n=len(gap_fillers),
    ax=ax
)
ax.figure.savefig('gap_fillers_heatmap.png', dpi=300, bbox_inches='tight')
```

**What We See**:
```
Gap fillers have moderate property values but occupy important positions:
- Bridge between high-sigma and low-kappa regions
- Fill the moderate band_gap space
- Connect different performance clusters
```

### Discovery Summary

**What We Found**:
1. **5 critical materials** at unique boundary positions (irreplaceable)
2. **10 clustered extremes** that can be represented by 3-4 samples
3. **8 gap fillers** that bridge property regions
4. **Key insight**: Property space has 5 distinct "corners" + bridging regions

**Research Strategy**:
- Synthesize analogs of the 5 critical materials (explore new boundaries)
- One synthesis per cluster of replaceable extremes (efficiency)
- Investigate gap fillers for intermediate applications
- **Saved**: ~50% of planned syntheses by identifying redundancy

---

## Example 3: Quality Control: Novel vs Known Variation

**Goal**: Distinguish truly unprecedented samples from known types of variation.

### The Dataset
```python
# Manufacturing sensor data (detecting defects)
sensor_data = pd.read_csv('manufacturing_sensors.csv')
print(f"Dataset: {sensor_data.shape[0]} measurements × {sensor_data.shape[1]} sensors")

# Historical data (training set) + New batch (test set)
historical = sensor_data[sensor_data['batch'] != 'new']
new_batch = sensor_data[sensor_data['batch'] == 'new']

print(f"Historical: {len(historical)} samples")
print(f"New batch: {len(new_batch)} samples")
```

### Step 1: Characterize Historical Variation
```python
# Fit on historical data
dt = DataTypical(
    shapley_mode=True,
    fast_mode=False
)

feature_cols = [c for c in sensor_data.columns if c.startswith('sensor_')]
X_historical = historical[feature_cols]
results_historical = dt.fit_transform(X_historical)

# Understand the structure of known variation
fig, ax = plt.subplots(figsize=(8, 6))
significance_plot(
    results_historical,
    significance='archetypal',
    ax=ax
)
ax.set_title('Historical Variation Structure', fontsize=16)
ax.figure.savefig('historical_structure.png', dpi=300, bbox_inches='tight')
```

### Step 2: Score New Batch
```python
# Transform new batch with fitted model
X_new = new_batch[feature_cols]
results_new = dt.transform(X_new)

# Compare to historical distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, sig in enumerate(['archetypal', 'prototypical', 'stereotypical']):
    axes[i].hist(results_historical[f'{sig}_rank'], bins=30, alpha=0.5,
                 label='Historical', color='#440154')   # viridis purple
    axes[i].hist(results_new[f'{sig}_rank'], bins=30, alpha=0.5,
                 label='New batch', color='#32b67a')    # viridis green
    axes[i].set_xlabel(f'{sig.capitalize()} Rank')
    axes[i].set_ylabel('Count')
    axes[i].legend(frameon=False)
    axes[i].set_title(f'{sig.capitalize()} Distribution')

plt.tight_layout()
fig.savefig('batch_comparison.png', dpi=300, bbox_inches='tight')
```

**What We See**:
```
Archetypal Distribution:
- Historical: Peak at 0.3, range 0.0-0.9
- New batch: Peak at 0.4, but 12 samples > 0.95

Interpretation: Most new samples are within known range, 
but 12 are MORE EXTREME than anything seen historically.
```

### Step 3: Investigate Unprecedented Samples
```python
# Flag samples exceeding historical boundaries
unprecedented = results_new[
    (results_new['archetypal_rank'] > results_historical['archetypal_rank'].max()) |
    (results_new['prototypical_rank'] < results_historical['prototypical_rank'].min())
]

print(f"\n{len(unprecedented)} unprecedented samples detected")

# Get formative ranks to understand if they're creating NEW structure
# Note: formative ranks from transform show how they'd fit into historical structure
unprecedented_idx = unprecedented.index

for idx in unprecedented_idx[:5]:  # Examine first 5
    print(f"\nSample {idx}:")
    print(f"  Archetypal: {results_new.loc[idx, 'archetypal_rank']:.3f}")
    print(f"  Prototypical: {results_new.loc[idx, 'prototypical_rank']:.3f}")
    
    # Get explanations
    explanations = dt.get_shapley_explanations(idx)
    arch_exp = explanations['archetypal']
    
    # Top contributing sensors
    top_sensors = np.argsort(np.abs(arch_exp))[-3:][::-1]
    print(f"  Top sensors: {[feature_cols[i] for i in top_sensors]}")
```

**Output**:
```
12 unprecedented samples detected

Sample 3421:
  Archetypal: 0.983
  Prototypical: 0.021
  Top sensors: ['sensor_07', 'sensor_23', 'sensor_14']

Sample 3445:
  Archetypal: 0.971
  Prototypical: 0.015
  Top sensors: ['sensor_23', 'sensor_07', 'sensor_31']

Pattern: sensor_07 and sensor_23 consistently appear
→ These sensors showing unprecedented variation
```

### Step 4: Profile Unprecedented Variation
```python
# Profile most extreme new sample
most_extreme = unprecedented['archetypal_rank'].idxmax()

fig, axes = plt.subplots(1, 2, figsize=(20, 5))

# Compare to most extreme historical
historical_extreme = results_historical['archetypal_rank'].idxmax()

profile_plot(dt, historical_extreme, significance='archetypal', 
             order='global', ax=axes[0])
axes[0].set_title('Most Extreme Historical Sample', fontsize=14)

profile_plot(dt, most_extreme, significance='archetypal', 
             order='global', ax=axes[1])
axes[1].set_title('Most Extreme New Sample', fontsize=14)

plt.tight_layout()
fig.savefig('unprecedented_comparison.png', dpi=300, bbox_inches='tight')
```

**What We See**:
```
Historical Extreme:
- Dominated by sensor_12 and sensor_19
- Moderate contributions from sensor_07

New Unprecedented:
- HUGE contribution from sensor_07 (3× larger than any historical)
- Large contribution from sensor_23 (never dominant historically)
- Small contribution from sensor_12

Interpretation: Entirely different failure mode!
Historical: Mechanical stress pattern (sensors 12, 19)
New: Thermal + electrical pattern (sensors 07, 23)
```

### Discovery Summary

**What We Found**:
1. **12 unprecedented samples** in new batch
2. **Novel failure mode** involving sensor_07 and sensor_23
3. **Pattern never seen** in historical data → requires investigation
4. **Quality control flag**: Hold batch, investigate thermal/electrical systems

**Action Taken**:
- Flagged batch for investigation
- Discovered cooling system malfunction (sensor_07 = temperature)
- Prevented shipping defective products
- **Value**: Caught issue that would have cost $$$$ in recalls

---

## Example 4: Biomarker Discovery: Pattern Exploration

**Goal**: Explore patient data to discover biomarker patterns without predefined hypotheses.

### The Dataset
```python
# Patient cohort with disease outcome
patients = pd.read_csv('patient_cohort.csv')
print(f"Dataset: {patients.shape[0]} patients × {patients.shape[1]} measurements")

# Biomarkers: gene expression, protein levels, metabolites
# Outcome: disease progression (0=stable, 1=progressive)
biomarker_cols = [c for c in patients.columns if c.startswith(('gene_', 'protein_', 'metab_'))]
print(f"Biomarkers: {len(biomarker_cols)}")
```

### Step 1: Exploratory Analysis
```python
# Analyze all three perspectives simultaneously
dt = DataTypical(
    shapley_mode=True,
    stereotype_column='outcome',  # Progressive vs stable
    fast_mode=False
)

X = patients[biomarker_cols]
results = dt.fit_transform(X)
patients_results = pd.concat([patients, results], axis=1)

# Overview: All three significance types
from datatypical_viz import plot_all_metrics

fig, axes = plot_all_metrics(
    results,
    color_by='outcome',
    figsize=(20, 6)
)
fig.savefig('all_perspectives.png', dpi=300, bbox_inches='tight')
```

**What We See**:
```
Archetypal (extreme combinations):
- Some progressive patients are extreme (unusual biomarker combinations)
- Some stable patients also extreme (different unusual combinations)

Prototypical (representative):
- Progressive and stable patients cluster separately
- Representatives of each outcome type clearly separated

Stereotypical (outcome-like):
- Strong separation of progressive-like vs stable-like
- Some stable patients are progressive-like (potential risk?)
```

### Step 2: Find Distinctive Biomarker Patterns
```python
# Heatmap of stereotypical (outcome-like) patients
# This shows biomarker patterns associated with progression

fig, ax = plt.subplots(figsize=(14, 10))
heatmap(
    dt, results,
    significance='stereotypical',
    order='actual',
    top_n=30,
    ax=ax
)
ax.figure.savefig('stereotypical_heatmap.png', dpi=300, bbox_inches='tight')
```

**What We See**:
```
Feature Patterns (global importance order):
1. protein_CRP (left-most) → Most important overall
2. gene_IL6
3. metab_glucose
4. protein_TNFa
5. gene_VEGF

Patient Clustering:
- Top 15 patients: All progressive, high protein_CRP, high gene_IL6
  → Inflammatory subtype
  
- Patients 16-25: Mixed outcomes, high metab_glucose, low protein_CRP
  → Metabolic subtype
  
- Patients 26-30: Stable patients with elevated gene_VEGF
  → Potentially at-risk group?
```

### Step 3: Investigate At-Risk Stable Patients
```python
# Find stable patients who are stereotypically progressive-like
at_risk = patients_results[
    (patients_results['outcome'] == 0) &  # Stable
    (patients_results['stereotypical_rank'] > 0.7)  # But progressive-like
]

print(f"\nFound {len(at_risk)} potentially at-risk stable patients")
print(at_risk[['patient_id', 'outcome', 'stereotypical_rank', 'age', 'follow_up_months']])

# Profile an at-risk patient
risk_idx = at_risk.index[0]
fig, ax = plt.subplots(figsize=(14, 6))
profile_plot(dt, risk_idx, significance='stereotypical', order='global', ax=ax)
ax.set_title(f'At-Risk Patient {patients.loc[risk_idx, "patient_id"]}', fontsize=16)
ax.figure.savefig('at_risk_profile.png', dpi=300, bbox_inches='tight')
```

**What We See**:
```
At-Risk Patient Profile:
- High gene_VEGF (largest positive contribution)
- Elevated protein_CRP (moderate positive)
- Low metab_NAD (large negative)
- Normal gene_IL6

Interpretation:
- Angiogenic + inflammatory signature (VEGF + CRP)
- But missing the IL6 inflammatory cascade
- Metabolic dysfunction (low NAD)

Hypothesis: Early-stage progressive pattern?
Recommendation: Close monitoring, follow-up at 3 months
```

### Step 4: Validate Subtypes
```python
# Compare inflammatory vs metabolic subtypes
# Top stereotypical with high formative = subtype-defining patients

inflammatory_subtype = patients_results[
    (patients_results['stereotypical_rank'] > 0.8) &
    (patients_results['stereotypical_shapley_rank'] > 0.7)
].head(5)

metabolic_subtype = patients_results[
    (patients_results['stereotypical_rank'] > 0.6) &
    (patients_results['stereotypical_rank'] < 0.8) &
    (patients_results['stereotypical_shapley_rank'] > 0.6)
].head(5)

# Profile one from each subtype
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

profile_plot(dt, inflammatory_subtype.index[0], 
             significance='stereotypical', order='global', ax=axes[0])
axes[0].set_title('Inflammatory Subtype Representative', fontsize=14)

profile_plot(dt, metabolic_subtype.index[0], 
             significance='stereotypical', order='global', ax=axes[1])
axes[1].set_title('Metabolic Subtype Representative', fontsize=14)

plt.tight_layout()
fig.savefig('subtype_comparison.png', dpi=300, bbox_inches='tight')
```

**What We See**:
```
Inflammatory Subtype:
- Dominated by protein_CRP, gene_IL6, protein_TNFa
- All inflammatory markers elevated
- Metabolic markers normal

Metabolic Subtype:
- Dominated by metab_glucose, metab_insulin, metab_NAD
- Inflammatory markers lower
- Metabolic dysfunction clear

Discovery: Two distinct pathways to disease progression!
```

### Discovery Summary

**What We Found**:
1. **Two disease subtypes** (inflammatory vs metabolic)
2. **18 at-risk stable patients** with progressive-like biomarkers
3. **Early-stage signature** (VEGF + low NAD) before full progression
4. **Biomarker panel**: CRP, IL6, glucose, NAD, VEGF distinguish subtypes

**Clinical Impact**:
- Stratify patients into subtypes for targeted treatment
- Monitor at-risk stable patients closely
- Develop subtype-specific biomarker panels
- **Value**: Personalized medicine approach from exploratory analysis

---

## Example 5: Dataset Curation: Coverage Gaps

**Goal**: Build a diverse, representative subset from a large dataset while identifying gaps.

### The Dataset
```python
# Large molecular database for machine learning
molecules = pd.read_csv('molecular_database.csv')  # 50,000 molecules
print(f"Full dataset: {molecules.shape[0]} molecules")

# Goal: Select 500 diverse molecules covering the space well
target_size = 500
```

### Step 1: Identify Structure
```python
# Analyze full dataset
dt = DataTypical(
    shapley_mode=True,
    fast_mode=True,  # Fast mode for large dataset
    shapley_top_n=1000  # Limit Shapley to top 1000
)

descriptor_cols = [c for c in molecules.columns if c.startswith('desc_')]
X = molecules[descriptor_cols]
results = dt.fit_transform(X)
molecules_results = pd.concat([molecules, results], axis=1)

# Visualize coverage structure
fig, ax = plt.subplots(figsize=(8, 6))
significance_plot(
    results,
    significance='prototypical',  # Representativeness
    ax=ax
)
ax.set_title('Coverage Structure (50K Molecules)', fontsize=16)
ax.figure.savefig('full_dataset_structure.png', dpi=300, bbox_inches='tight')
```

### Step 2: Select Diverse Representatives
```python
# Strategy: Select based on dual perspective
# 1. High prototypical + high formative = critical representatives
# 2. High archetypal + high formative = boundary definers
# 3. Low prototypical + high formative = gap fillers

# Critical representatives (top-right in prototypical plot)
critical_reps = molecules_results[
    (molecules_results['prototypical_rank'] > 0.7) &
    (molecules_results['prototypical_shapley_rank'] > 0.7)
]

# Boundary definers (top-right in archetypal plot)
boundary_definers = molecules_results[
    (molecules_results['archetypal_rank'] > 0.9) &
    (molecules_results['archetypal_shapley_rank'] > 0.7)
]

# Gap fillers (top-left in prototypical plot)
gap_fillers = molecules_results[
    (molecules_results['prototypical_rank'] < 0.4) &
    (molecules_results['prototypical_shapley_rank'] > 0.6)
]

print(f"Critical representatives: {len(critical_reps)}")
print(f"Boundary definers: {len(boundary_definers)}")
print(f"Gap fillers: {len(gap_fillers)}")
```

**Output**:
```
Critical representatives: 234
Boundary definers: 89
Gap fillers: 143
Total: 466 molecules
```

### Step 3: Fill to Target Size
```python
# Still need 500 - 466 = 34 more molecules
# Use highest formative from remaining molecules

remaining = molecules_results[
    ~molecules_results.index.isin(
        critical_reps.index.union(boundary_definers.index).union(gap_fillers.index)
    )
]

additional = remaining.nlargest(34, 'prototypical_shapley_rank')

# Combine all selected molecules
selected = pd.concat([critical_reps, boundary_definers, gap_fillers, additional])
print(f"\nFinal curated set: {len(selected)} molecules")

# Save curated dataset
selected.to_csv('curated_molecules_500.csv', index=False)
```

### Step 4: Validate Coverage
```python
# Fit on curated set and compare to full set
dt_curated = DataTypical(shapley_mode=False, fast_mode=True)
results_curated = dt_curated.fit_transform(selected[descriptor_cols])

# Compare archetypal ranges (coverage of extreme space)
full_arch_range = (
    results['archetypal_rank'].max() - results['archetypal_rank'].min()
)
curated_arch_range = (
    results_curated['archetypal_rank'].max() - results_curated['archetypal_rank'].min()
)

print(f"\nArchetypal range:")
print(f"  Full dataset: {full_arch_range:.3f}")
print(f"  Curated set: {curated_arch_range:.3f}")
print(f"  Coverage: {(curated_arch_range/full_arch_range)*100:.1f}%")

# Compare prototypical distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(results['prototypical_rank'], bins=50, alpha=0.7, color='#440154')
axes[0].set_xlabel('Prototypical Rank')
axes[0].set_ylabel('Count')
axes[0].set_title('Full Dataset (50K molecules)')

axes[1].hist(results_curated['prototypical_rank'], bins=30, alpha=0.7, color='#32b67a')
axes[1].set_xlabel('Prototypical Rank')
axes[1].set_ylabel('Count')
axes[1].set_title('Curated Set (500 molecules)')

plt.tight_layout()
fig.savefig('coverage_validation.png', dpi=300, bbox_inches='tight')
```

**Output**:
```
Archetypal range:
  Full dataset: 0.987
  Curated set: 0.981
  Coverage: 99.4%

Prototypical distribution:
  Full: Heavy center, sparse boundaries
  Curated: Flatter distribution, better boundary coverage
  
Interpretation: Curated set captures 99% of extreme space
with 1% of molecules, and has BETTER boundary coverage
than random sampling.
```

### Step 5: Identify Remaining Gaps
```python
# Transform full dataset with curated model to find gaps
results_full_with_curated = dt_curated.transform(X)

# Find molecules far from curated representatives
gaps = molecules_results[
    (results_full_with_curated['prototypical_rank'] < 0.2)  # Far from representatives
].nlargest(20, 'prototypical_shapley_rank')  # But would be important

print(f"\nIdentified {len(gaps)} high-priority gaps")
print("These regions need more sampling:")
print(gaps[['molecule_id', 'prototypical_rank', 'prototypical_shapley_rank']])

# Profile a gap region
gap_idx = gaps.index[0]
fig, ax = plt.subplots(figsize=(14, 6))
profile_plot(dt_curated, gap_idx, significance='prototypical', order='global', ax=ax)
ax.set_title('Identified Gap Region', fontsize=16)
ax.figure.savefig('gap_profile.png', dpi=300, bbox_inches='tight')
```

**What We See**:
```
Gap region characteristics:
- High desc_polar (largest contribution)
- High desc_flexible
- Low desc_aromatic
- Moderate desc_MW

Interpretation: Undersampled region = polar, flexible, non-aromatic
Recommendation: Prioritize synthesis/acquisition in this chemical space
```

### Discovery Summary

**What We Accomplished**:
1. **Curated 500 molecules** from 50,000 (1% of data)
2. **99.4% coverage** of extreme property space
3. **Better boundary coverage** than random sampling
4. **Identified 20 gap regions** for future sampling priority

**Dataset Quality**:
- 234 critical representatives (core structure)
- 89 boundary definers (extreme coverage)
- 143 gap fillers (complete coverage)
- 34 additional structure contributors

**Impact**:
- Machine learning models trained on curated set perform within 2% of full dataset
- Training time reduced 100× (minutes vs hours)
- Identified specific regions needing more data
- **Value**: Efficient, high-quality dataset for downstream tasks

---

## Summary: Common Patterns Across Examples

### What DataTypical Enables

1. **Mechanism Discovery** (Example 1)
   - Multiple pathways to same outcome
   - Feature patterns distinguish mechanisms
   - Formative instances define unique approaches

2. **Structural Understanding** (Example 2)
   - Boundary extremes vs clustered extremes
   - Gap-filling instances bridge regions
   - Critical vs replaceable samples

3. **Quality Control** (Example 3)
   - Novel vs known variation
   - Unprecedented patterns flagged
   - Root cause through feature attribution

4. **Pattern Exploration** (Example 4)
   - Subtype discovery without hypotheses
   - At-risk identification through stereotypical analysis
   - Biomarker signatures emerge naturally

5. **Dataset Curation** (Example 5)
   - Intelligent subset selection
   - Coverage validation
   - Gap identification for future sampling

### The Dual Perspective Value

In every example, the distinction between **actual** (what IS) and **formative** (what CREATES) revealed insights impossible from ranks alone:

- Drug discovery: Active compounds that were redundant vs unique
- Materials: Extremes that were clustered vs boundary-defining
- Quality control: Known variation vs unprecedented failure modes
- Biomarkers: Representative patterns vs subtype-defining patterns
- Curation: Representative molecules vs structure-defining molecules

### Workflow Template

1. **Explore** quickly (fast_mode=True, no Shapley)
2. **Identify** interesting regions/samples
3. **Understand** with explanations (shapley_mode=True, subset)
4. **Discover** patterns through visualization
5. **Act** on insights (synthesis, monitoring, curation, etc.)

---

**Ready to apply these workflows to your data?**

See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for parameter details and [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) for interpretation guidance.