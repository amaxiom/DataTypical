"""
DataTypical v0.7 - Visualization Module
========================================

Publication-quality visualizations for dual-perspective analysis:
1. significance_plot: Dual-perspective scatter (hero visualization)
   - Automatic discrete/continuous color detection
   - Binary: purple (low) and green (high)
   - Discrete (3-12 categories): viridis palette + legend
   - Continuous (>12 values): viridis colormap + colorbar
2. heatmap: Feature attribution heatmaps
3. profile_plot: Feature importance profiles for individual samples

Design specifications:
- Viridis colormap (default)
- figsize (6,5) for scatter and heatmaps
- figsize (12,5) for profile plots
- Font size 12 for ticks, 14 for axis labels
- Clean, professional style matching existing software

Author: Amanda S. Barnard
Date: January 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Optional, Dict, List, Tuple, Union

# Configure plotting defaults
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12


# ============================================================================
# HELPER function
# ============================================================================

def get_top_sample(
    results: pd.DataFrame,
    rank_column: str,
    n: int = 1,
    mode: str = 'max'
) -> Union[int, List[int], None]:
    """
    Safely get top sample(s) from results, handling missing formative data.
    
    Parameters
    ----------
    results : pd.DataFrame
        Results from DataTypical.fit_transform()
    rank_column : str
        Column to rank by (e.g., 'archetypal_rank', 'archetypal_shapley_rank')
    n : int
        Number of top samples to return (default: 1)
    mode : str
        'max' for highest values, 'min' for lowest (default: 'max')
    
    Returns
    -------
    sample_idx : int, list, or None
        Top sample index/indices, or None if data not available
        Returns single int if n=1, list if n>1
    
    Examples
    --------
    >>> # Get top archetypal sample
    >>> top_idx = get_top_sample(results, 'archetypal_rank')
    >>> 
    >>> # Get top formative sample (handles NaN gracefully)
    >>> top_formative = get_top_sample(results, 'archetypal_shapley_rank')
    >>> if top_formative is not None:
    ...     ax = profile_plot(dt, top_formative, order='global')
    >>> 
    >>> # Get top 5 samples
    >>> top_5 = get_top_sample(results, 'prototypical_rank', n=5)
    """
    if rank_column not in results.columns:
        print(f"âš  Column '{rank_column}' not found in results")
        return None
    
    # Check if column is all NaN (formative data not available)
    if results[rank_column].isna().all():
        print(f"âš  Column '{rank_column}' has no data (likely fast_mode=True)")
        
        # Provide helpful message based on column name
        if 'shapley_rank' in rank_column:
            print(f"  Formative data requires: DataTypical(shapley_mode=True, fast_mode=False)")
            if 'stereotypical' in rank_column:
                print(f"  Also requires: stereotype_column='<your_column>'")
        
        return None
    
    # Get top sample(s)
    if mode == 'max':
        if n == 1:
            return results[rank_column].idxmax()
        else:
            return results.nlargest(n, rank_column).index.tolist()
    else:  # min
        if n == 1:
            return results[rank_column].idxmin()
        else:
            return results.nsmallest(n, rank_column).index.tolist()


# ============================================================================
# HERO VISUALIZATION: Dual-Perspective Scatter
# ============================================================================

def significance_plot(
    results: pd.DataFrame,
    significance: str = 'archetypal',
    color_by: Optional[str] = None,
    size_by: Optional[str] = None,
    labels: Optional[pd.Series] = None,
    label_top: int = 0,
    quadrant_lines: bool = True,
    quadrant_threshold: Tuple[float, float] = (0.5, 0.5),
    figsize: Tuple[int, int] = (6, 5),
    cmap: str = 'viridis',
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Axes:
    """
    Create dual-perspective significance scatter plot (hero visualization).
    
    This plot reveals the relationship between actual significance (samples that
    ARE archetypal/prototypical/stereotypical) and formative significance 
    (samples that CREATE the structure).
    
    Parameters
    ----------
    results : pd.DataFrame
        Results from DataTypical.fit_transform() with shapley_mode=True
    significance : str
        Which significance to plot: 'archetypal', 'prototypical', or 'stereotypical'
        Automatically uses {significance}_rank (x-axis) and {significance}_shapley_rank (y-axis)
    color_by : str, optional
        Column name to color points by. Automatically detects:
        - Continuous variables (>12 unique values): colored with viridis colormap + colorbar
        - Binary variables (2 unique values): purple (low) and green (high) + legend
        - Discrete variables (3-5 unique): discrete viridis colors + legend
        - Discrete variables (6-12 unique): discrete colors + different markers + legend
        - Non-numeric variables: discrete viridis colors + legend
    size_by : str, optional
        Column name to size points by
    labels : pd.Series, optional
        Labels for points (e.g., compound IDs)
    label_top : int
        Number of top points to label (by actual+formative)
    quadrant_lines : bool
        Draw quadrant division lines
    quadrant_threshold : tuple of float
        (x, y) thresholds for quadrant lines
    figsize : tuple of int
        Figure size (width, height)
    cmap : str
        Colormap name (default: 'viridis')
    title : str, optional
        Plot title
    ax : plt.Axes, optional
        Existing axes to plot on
    **kwargs
        Additional arguments passed to scatter()
    
    Returns
    -------
    ax : plt.Axes
        Matplotlib axes object
    
    Examples
    --------
    >>> dt = DataTypical(shapley_mode=True)
    >>> results = dt.fit_transform(data)
    >>> 
    >>> # Basic archetypal plot
    >>> ax = significance_plot(results, significance='archetypal')
    >>> 
    >>> # Prototypical with continuous color
    >>> ax = significance_plot(
    ...     results, 
    ...     significance='prototypical',
    ...     color_by='solubility',  # continuous variable
    ...     label_top=5
    ... )
    >>> 
    >>> # Archetypal with binary color (auto-detected)
    >>> ax = significance_plot(
    ...     results,
    ...     significance='archetypal',
    ...     color_by='treatment'  # binary: 0/1 or control/treatment
    ... )
    >>> 
    >>> # Stereotypical with discrete multi-class color
    >>> ax = significance_plot(
    ...     results,
    ...     significance='stereotypical',
    ...     color_by='cell_type'  # 4 categories with different colors
    ... )
    """
    
    # Validate significance parameter
    valid_significance = ['archetypal', 'prototypical', 'stereotypical']
    if significance not in valid_significance:
        raise ValueError(f"significance must be one of {valid_significance}, got '{significance}'")
    
    # Auto-determine column names
    actual_col = f'{significance}_rank'
    formative_col = f'{significance}_shapley_rank'
    
    if actual_col not in results.columns:
        raise ValueError(f"Column '{actual_col}' not in results")
    if formative_col not in results.columns:
        raise ValueError(f"Column '{formative_col}' not in results")
    
    # Check if formative data is available (could be None in fast_mode)
    if results[formative_col].isna().all():
        # Print informative message and skip plot
        print(f"\nâš  Skipping significance plot:")
        print(f"  Formative data ('{formative_col}') not available (fast_mode=True)")
        print(f"  This plot requires fast_mode=False to compute formative Shapley values")
        
        # Return None or empty axes depending on whether axes was provided
        if ax is None:
            # Create empty figure with message
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'Formative data not available\n(fast_mode=True)', 
                   ha='center', va='center', fontsize=14, color='gray')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel(f'{significance.capitalize()} Rank (Actual)', fontsize=14)
            ax.set_ylabel(f'{significance.capitalize()} Rank (Formative)', fontsize=14)
            if title:
                ax.set_title(title, fontsize=14)
            return ax
        return ax
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare size values (needed for all plot types)
    if size_by is not None:
        if size_by not in results.columns:
            raise ValueError(f"size_by column '{size_by}' not found in results")
        s_values = results[size_by]
        # Normalize to reasonable range (10-200)
        s_min, s_max = s_values.min(), s_values.max()
        if s_max > s_min:
            s_normalized = 10 + 190 * (s_values - s_min) / (s_max - s_min)
        else:
            s_normalized = 50
    else:
        s_normalized = 50
    
    # Handle coloring logic with automatic discrete/continuous detection
    if color_by is not None:
        if color_by not in results.columns:
            raise ValueError(f"color_by column '{color_by}' not found in results")
        
        c_values = results[color_by]
        
        # Detect variable type: discrete vs continuous
        is_numeric = pd.api.types.is_numeric_dtype(c_values)
        n_unique = c_values.nunique()
        
        # Determine if discrete or continuous
        if not is_numeric:
            # Non-numeric → discrete
            is_discrete = True
            use_markers = False
        elif n_unique < 6:
            # Numeric with <6 values → discrete
            is_discrete = True
            use_markers = False
        elif 6 <= n_unique <= 12:
            # Numeric with 6-12 values → discrete with markers
            is_discrete = True
            use_markers = True
        else:
            # Numeric with >12 values → continuous
            is_discrete = False
            use_markers = False
        
        if is_discrete:
            # DISCRETE COLORING: Use legend instead of colorbar
            
            # Get unique values in sorted order for consistent mapping
            if is_numeric:
                unique_values = sorted(c_values.dropna().unique())
            else:
                unique_values = sorted(c_values.dropna().unique(), key=str)
            
            n_categories = len(unique_values)
            
            if n_categories == 0:
                raise ValueError(f"color_by column '{color_by}' has no valid values")
            
            # Create color mapping
            if n_categories == 2:
                # BINARY: purple (viridis[0.0]) for lower, green (viridis[0.6]) for higher
                viridis_cmap = plt.cm.get_cmap('viridis')
                color_map = {
                    unique_values[0]: viridis_cmap(0.0),     # purple for lower value
                    unique_values[1]: viridis_cmap(0.66)     # green for higher value
                }
            else:
                # MULTI-CLASS: discrete viridis palette
                viridis_cmap = plt.cm.get_cmap('viridis', n_categories)
                color_map = {val: viridis_cmap(i) for i, val in enumerate(unique_values)}
            
            # Create marker mapping if needed
            if use_markers:
                # 6-12 categories: use different markers for each category
                marker_list = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', 'P', '<', '>']
                marker_map = {val: marker_list[i % len(marker_list)] 
                            for i, val in enumerate(unique_values)}
            
            # Plot each category separately for legend
            for category in unique_values:
                mask = c_values == category
                
                # Get marker for this category
                if use_markers:
                    marker = marker_map[category]
                else:
                    marker = 'o'
                
                # Get size values for this subset
                if isinstance(s_normalized, (int, float)):
                    s_subset = s_normalized
                else:
                    s_subset = s_normalized[mask]
                
                # Plot this category
                ax.scatter(
                    results.loc[mask, actual_col],
                    results.loc[mask, formative_col],
                    c=[color_map[category]],
                    s=s_subset,
                    marker=marker,
                    edgecolors='black',
                    linewidth=0.5,
                    label=str(category),
                    **kwargs
                )
            
            # Add legend
            ax.legend(
                title=color_by,
                fontsize=12,
                title_fontsize=12,
                loc='center left',
                bbox_to_anchor=(1.0, 0.5),
                frameon=False
            )
            
        else:
            # CONTINUOUS COLORING: Use colorbar (existing behavior)
            scatter = ax.scatter(
                results[actual_col],
                results[formative_col],
                c=c_values,
                s=s_normalized,
                cmap=cmap,
                edgecolors='black',
                linewidth=0.5,
                **kwargs
            )
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(color_by, fontsize=14)
            cbar.ax.tick_params(labelsize=12)
    
    else:
        # NO COLORING: Use default steelblue
        ax.scatter(
            results[actual_col],
            results[formative_col],
            c='steelblue',
            s=s_normalized,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5,
            **kwargs
        )
    
    # Add quadrant lines if requested
    if quadrant_lines:
        x_thresh, y_thresh = quadrant_threshold
        ax.axvline(x_thresh, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y_thresh, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Label top points if requested
    if label_top > 0 and labels is not None:
        # Rank by sum of actual + formative
        combined_rank = results[actual_col] + results[formative_col]
        top_indices = combined_rank.nlargest(label_top).index
        
        for idx in top_indices:
            ax.annotate(
                labels[idx],
                xy=(results.loc[idx, actual_col], results.loc[idx, formative_col]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='gray', lw=0.5)
            )
    
    # Labels and title
    ax.set_xlabel(f'{significance.capitalize()} Rank (Actual)', fontsize=14)
    ax.set_ylabel(f'{significance.capitalize()} Rank (Formative)', fontsize=14)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'{significance.capitalize()} Dual-Perspective Analysis', fontsize=14)
    
    # Grid for readability
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Ensure limits include full range
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    return ax


# ============================================================================
# FEATURE ATTRIBUTION HEATMAP
# ============================================================================

def heatmap(
    dt_fitted,
    results: Optional[pd.DataFrame] = None,
    samples: Optional[Union[List[int], np.ndarray]] = None,
    significance: str = 'archetypal',
    order: str = 'actual',
    top_n: Optional[int] = None,
    top_features: int = None,
    figsize: Tuple[int, int] = (6, 5),
    cmap: str = 'viridis',
    center: Optional[float] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Axes:
    """
    Create feature attribution heatmap for Shapley explanations.
    
    Always shows explanations (why samples ARE significant). Features are always
    ordered by global importance (average across all samples). Instances can be
    ordered by actual or formative ranks.
    
    Parameters
    ----------
    dt_fitted : DataTypical
        Fitted DataTypical instance with shapley_mode=True
    results : pd.DataFrame, optional
        Results dataframe from fit_transform(). Required if samples not specified.
    samples : list or array, optional
        Indices of samples to show. If None, uses top_n samples from results.
    significance : str
        Which significance to visualize: 'archetypal', 'prototypical', 'stereotypical'
    order : str
        How to order instances (rows): 'actual' or 'formative'
        'actual': Order by {significance}_rank (samples that ARE significant)
        'formative': Order by {significance}_shapley_rank (samples that CREATE structure)
    top_n : int, optional
        Number of top samples to show (if samples not specified).
        Defaults to shapley_top_n from fitted model, or 20 if not set.
    top_features : int, optional
        Number of top features to show (by absolute attribution)
    figsize : tuple of int
        Figure size (width, height)
    cmap : str
        Colormap name (default: 'viridis')
    center : float, optional
        Value to center colormap at (for diverging colormaps)
    title : str, optional
        Plot title
    ax : plt.Axes, optional
        Existing axes to plot on
    **kwargs
        Additional arguments passed to seaborn.heatmap()
    
    Returns
    -------
    ax : plt.Axes
        Matplotlib axes object
    
    Examples
    --------
    >>> dt = DataTypical(shapley_mode=True)
    >>> results = dt.fit_transform(data)
    >>> 
    >>> # Show explanations for top 10 archetypal samples
    >>> ax = heatmap(dt, results, top_n=10, significance='archetypal')
    >>> 
    >>> # Order by formative ranks
    >>> ax = heatmap(dt, results, order='formative', significance='archetypal')
    >>> 
    >>> # Show top 20 prototypical with only top 15 features
    >>> ax = heatmap(dt, results, significance='prototypical', top_n=20, top_features=15)
    """
    
    if not dt_fitted.shapley_mode:
        raise RuntimeError("Shapley mode not enabled. Set shapley_mode=True when fitting.")
    
    # Default top_n to shapley_top_n from fitted model
    if top_n is None:
        if hasattr(dt_fitted, 'shapley_top_n') and dt_fitted.shapley_top_n is not None:
            # Convert fraction to absolute count if needed
            if isinstance(dt_fitted.shapley_top_n, float) and 0 < dt_fitted.shapley_top_n < 1:
                n_samples = len(dt_fitted.train_index_) if dt_fitted.train_index_ is not None else 100
                top_n = max(1, int(dt_fitted.shapley_top_n * n_samples))
            else:
                top_n = int(dt_fitted.shapley_top_n)
        else:
            top_n = 20  # Fallback default
    
    # Get explanations matrix (always use explanations, not formative)
    if significance == 'archetypal':
        Phi = dt_fitted.Phi_archetypal_explanations_
    elif significance == 'prototypical':
        Phi = dt_fitted.Phi_prototypical_explanations_
    elif significance == 'stereotypical':
        Phi = dt_fitted.Phi_stereotypical_explanations_
    else:
        raise ValueError(f"Unknown significance: {significance}")
    
    # Check if data is available
    if Phi is None:
        print(f"\n⚠ Skipping {significance} explanations heatmap:")
        print(f"  Explanations data not available")
        
        if significance == 'stereotypical':
            print(f"  Note: Stereotypical also requires stereotype_column to be set")
        
        print(f"\n  To enable this plot, refit with:")
        if significance == 'stereotypical':
            print(f"    DataTypical(shapley_mode=True, stereotype_column='<column>')")
        else:
            print(f"    DataTypical(shapley_mode=True)")
        
        # Return empty axes with message
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        ax.text(0.5, 0.5, f'{significance.capitalize()} explanations\nnot available', 
               ha='center', va='center', fontsize=14, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        return ax
    
    # Select samples
    if samples is None:
        if results is None:
            raise ValueError("Must provide either 'samples' or 'results' dataframe")
        
        # Determine rank column based on order
        if order == 'actual':
            rank_col = f"{significance}_rank"
        elif order == 'formative':
            rank_col = f"{significance}_shapley_rank"
            
            # Check if formative data is available
            if results[rank_col].isna().all():
                print(f"\n⚠ Warning: order='formative' requested but formative data not available")
                print(f"  Falling back to ordes='actual'")
                rank_col = f"{significance}_rank"
                order = 'actual'
        else:
            raise ValueError(f"order must be 'actual' or 'formative', got '{order}'")
        
        if rank_col not in results.columns:
            raise RuntimeError(f"Cannot find {rank_col} in results")
        
        # Get DataFrame indices of top samples (already ordered by rank)
        top_samples_df_indices = results.nlargest(top_n, rank_col).index
        
        # FIX: Use get_shapley_explanations to handle shapley_top_n subsampling
        Phi_subset_list = []
        sample_labels = []
        
        for df_idx in top_samples_df_indices:
            try:
                # Get explanations using the API (handles index mapping internally)
                explanations = dt_fitted.get_shapley_explanations(df_idx)
                shapley_values = explanations[significance]
                
                # Only include if we have valid data
                if np.any(shapley_values != 0) or not hasattr(dt_fitted, 'shapley_top_n') or dt_fitted.shapley_top_n is None:
                    Phi_subset_list.append(shapley_values)
                    sample_labels.append(str(df_idx))
            except (IndexError, KeyError):
                # This sample doesn't have explanations computed
                pass
        
        if len(Phi_subset_list) == 0:
            print(f"\n⚠ Error: None of the top {top_n} {significance} instances have explanations")
            print(f"  This can happen when shapley_top_n is too small")
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No explanations available\nfor top {top_n} instances', 
                   ha='center', va='center', fontsize=14, color='gray')
            ax.axis('off')
            return ax
        
        Phi_subset = np.array(Phi_subset_list)
        
        # WARNING: Check for zero Shapley values when using formative ordering
        if order == 'formative':
            zero_count = (Phi_subset == 0).all(axis=1).sum()
            if zero_count > 0:
                print(f"\n⚠ Warning: {zero_count}/{len(sample_labels)} top formative {significance} instances have zero Shapley values")
                print(f"  This occurs when a formative instance is not in the top {significance} instances")
                print(f"  (determined by shapley_top_n parameter)")
                print(f"  These instances CREATE structure but are not themselves highly {significance}")
        
    else:
        # Custom sample list provided - convert to positional indices
        samples = np.asarray(samples)
        
        # Extract Shapley values for selected samples
        # CRITICAL: Keep rank order - DO NOT re-sort!
        # Top-ranked sample should appear at TOP of heatmap
        Phi_subset = Phi[samples, :]
        
        # Create DataFrame for heatmap with actual sample IDs
        if dt_fitted.train_index_ is not None:
            # Use actual DataFrame indices from training
            sample_labels = [str(dt_fitted.train_index_[s]) for s in samples]
        else:
            sample_labels = [f"Sample {s}" for s in samples]
    
    # Get feature names
    feature_names = [dt_fitted.feature_columns_[i] for i, keep in enumerate(dt_fitted.keep_mask_) if keep]
    
    # ALWAYS order features by global importance (average across ALL samples)
    global_importance = np.abs(Phi).mean(axis=0)
    feature_order = np.argsort(global_importance)[::-1]
    Phi_subset = Phi_subset[:, feature_order]
    feature_names = [feature_names[i] for i in feature_order]
    
    # Select top features if requested
    if top_features is not None:
        Phi_subset = Phi_subset[:, :top_features]
        feature_names = feature_names[:top_features]
    
    # Create DataFrame for heatmap
    df_heatmap = pd.DataFrame(
        Phi_subset,
        index=sample_labels,
        columns=feature_names
    )
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    heatmap_kwargs = {
        'cmap': cmap,
        'center': center,
        'cbar_kws': {'label': 'Shapley Value', 'shrink': 0.5},
        'linewidths': 0.5,
        'annot': False,
        'fmt': '.3f',
        'square': False
    }
    heatmap_kwargs.update(kwargs)
    
    sns.heatmap(df_heatmap, ax=ax, **heatmap_kwargs)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor('black')
    cbar = ax.collections[0].colorbar
    cbar.outline.set_linewidth(1)
    cbar.outline.set_edgecolor('black')
    
    # Labels
    ax.set_xlabel('Features (Ordered by Global Importance)', fontsize=12)
    if order == 'actual':
        ylabel = f'Samples (Ordered by {significance.title()} Rank)'
    else:
        ylabel = f'Samples (Ordered by Formative {significance.title()} Rank)'
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='x', labelrotation=90, labelsize=12)
    ax.tick_params(axis='y', labelrotation=0, labelsize=12)
   
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'{significance.title()} Explanations', fontsize=14)
    
    plt.tight_layout()
    
    return ax

# ============================================================================
# INDIVIDUAL SAMPLE PROFILE
# ============================================================================

def profile_plot(
    dt_fitted,
    sample_idx: int,
    significance: str = 'archetypal',
    order: str = 'local',
    top_features: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 5),
    cmap: str = 'viridis',
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Axes:
    """
    Create feature importance profile for a single sample.
    
    Shows Shapley explanations (y-axis) for each feature (x-axis), with bars colored by
    the normalized feature value for this sample. Features can be ordered locally (by
    this sample's importance) or globally (by average importance across all samples).
    
    Missingness Indicator: Features that had missing values in the original data
    appear as colorless (transparent) bars, while observed features are colored. This
    preserves the signal that imputation was used and distinguishes real from imputed data.
    Bar height (Shapley value) is always shown regardless of missingness.
    
    Parameters
    ----------
    dt_fitted : DataTypical
        Fitted DataTypical instance with shapley_mode=True
    sample_idx : int
        Index of sample to profile
    significance : str
        Which significance to visualize: 'archetypal', 'prototypical', 'stereotypical'
    order : str
        Feature ordering method: 'local' or 'global'
        'local': Order by this sample's Shapley values
        'global': Order by average importance across all samples (uses explanations)
    top_features : int, optional
        Number of top features to display. If None, shows all features.
        Features are selected after ordering (top N by importance).
    figsize : tuple of int
        Figure size (width, height)
    cmap : str
        Colormap name (default: 'viridis')
    title : str, optional
        Plot title
    ax : plt.Axes, optional
        Existing axes to plot on
    **kwargs
        Additional arguments passed to bar()
    
    Returns
    -------
    ax : plt.Axes
        Matplotlib axes object
    
    Examples
    --------
    >>> dt = DataTypical(shapley_mode=True)
    >>> dt.fit(data)
    >>> 
    >>> # Profile top archetypal sample (local ordering)
    >>> top_idx = results['archetypal_rank'].idxmax()
    >>> ax = profile_plot(dt, top_idx, significance='archetypal', order='local')
    >>> 
    >>> # Profile top formative sample (global ordering)
    >>> top_formative = results['archetypal_shapley_rank'].idxmax()
    >>> ax = profile_plot(dt, top_formative, significance='archetypal', order='global')
    >>> 
    >>> # Show only top 15 most important features
    >>> ax = profile_plot(dt, top_idx, significance='archetypal', 
    ...                   order='local', top_features=15)
    """
    
    if not dt_fitted.shapley_mode:
        raise RuntimeError("Shapley mode not enabled. Set shapley_mode=True when fitting.")
    
    # Validate parameters
    valid_significance = ['archetypal', 'prototypical', 'stereotypical']
    if significance not in valid_significance:
        raise ValueError(f"significance must be one of {valid_significance}, got '{significance}'")
    
    valid_order = ['local', 'global']
    if order not in valid_order:
        raise ValueError(f"order must be one of {valid_order}, got '{order}'")
    
    # Get explanations for this sample
    explanations = dt_fitted.get_shapley_explanations(sample_idx)
    shapley_values = explanations[significance]
    
    # Get feature names
    feature_names = [dt_fitted.feature_columns_[i] for i, keep in enumerate(dt_fitted.keep_mask_) if keep]
    
    # Determine feature ordering
    if order == 'local':
        # Order by THIS sample's Shapley values
        importance = np.abs(shapley_values)
        ordering_type = "local"
    else:  # global
        # Order by average importance across ALL samples (using explanations)
        if significance == 'archetypal':
            Phi_explanations = dt_fitted.Phi_archetypal_explanations_
        elif significance == 'prototypical':
            Phi_explanations = dt_fitted.Phi_prototypical_explanations_
        else:  # stereotypical
            Phi_explanations = dt_fitted.Phi_stereotypical_explanations_
        
        # Calculate global importance (average across all samples)
        importance = np.mean(np.abs(Phi_explanations), axis=0)
        ordering_type = "global"
    
    # Sort features by importance (descending by absolute value)
    sorted_idx = np.argsort(importance)[::-1]
    
    # Get sorted Shapley values (keep original signs for plotting)
    shapley_sorted = shapley_values[sorted_idx]
    features_sorted = [feature_names[i] for i in sorted_idx]
    
    # Get normalized feature values for coloring
    # Need to get the actual feature values for this sample
    original_data = dt_fitted._df_original_fit
    numeric_cols = [dt_fitted.feature_columns_[i] for i, keep in enumerate(dt_fitted.keep_mask_) if keep]
    
    # Get sample's feature values
    sample_feature_values = original_data.loc[sample_idx, numeric_cols].values.astype(np.float64)
    
    # Get dataset min/max for normalization
    dataset_values = original_data[numeric_cols].values.astype(np.float64)
    feat_min = dataset_values.min(axis=0)
    feat_max = dataset_values.max(axis=0)
    
    # Normalize to [0, 1]
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0  # Avoid division by zero
    normalized_values = (sample_feature_values - feat_min) / feat_range
    
    # Sort by same order as Shapley values
    normalized_sorted = normalized_values[sorted_idx]
    
    # Apply top_features filter if specified
    if top_features is not None:
        if top_features < 1:
            raise ValueError(f"top_features must be >= 1, got {top_features}")
        
        n_features = len(features_sorted)
        if top_features < n_features:
            # Slice to keep only top N features
            shapley_sorted = shapley_sorted[:top_features]
            features_sorted = features_sorted[:top_features]
            normalized_sorted = normalized_sorted[:top_features]
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Create color array from normalized feature values
    colors = plt.cm.get_cmap(cmap)(normalized_sorted)
    
    # Create bar plot - use signed Shapley values (can be negative)
    x_pos = np.arange(len(features_sorted))
    bars = ax.bar(x_pos, shapley_sorted, color=colors, edgecolor='black', linewidth=0.5, **kwargs)
    
    # Add zero reference line
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Normalized Feature Value', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    # Labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(features_sorted, rotation=90, ha='center', fontsize=12)
    
    # Update xlabel based on ordering type and top_features
    if top_features is not None:
        feature_label = f'Top {top_features} Features (Ordered by {ordering_type.capitalize()} Importance)'
    else:
        feature_label = f'Features (Ordered by {ordering_type.capitalize()} Importance)'
    
    ax.set_xlabel(feature_label, fontsize=14)
    ax.set_ylabel('Shapley Value', fontsize=14)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        # Get sample label if available
        sample_label = f"Sample {sample_idx}"
        if hasattr(dt_fitted, 'label_df_') and dt_fitted.label_df_ is not None:
            if len(dt_fitted.label_df_.columns) > 0:
                first_label_col = dt_fitted.label_df_.columns[0]
                if sample_idx in dt_fitted.label_df_.index:
                    sample_label = dt_fitted.label_df_.loc[sample_idx, first_label_col]
        
        # Add significance to title
        sig_name = significance.capitalize()
        ax.set_title(f'{sig_name} Explanations: {sample_idx}', fontsize=14)
    
    # Grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    return ax

# ============================================================================
# Export
# ============================================================================

__all__ = [
    'significance_plot',
    'heatmap', 
    'profile_plot',
    'get_top_sample'  
]
