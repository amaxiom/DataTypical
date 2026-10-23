"""
DataTypical v0.8.2 - Visualization Module
========================================

Publication-quality visualizations for dual-perspective analysis:
1. significance_plot: Dual-perspective scatter (hero visualization)
   - Automatic discrete/continuous color detection
   - Binary: purple (low) and green (high)
   - Discrete (3-12 categories): viridis palette + legend
   - Continuous (>12 values): viridis colormap + colorbar
2. heatmap: Feature attribution heatmaps
3. profile_plot: Feature importance profiles for individual samples

All three serve tabular, text and graph fits. A graph fit is a tabular fit of
the node features plus any topology columns; a text fit names its features by
the fitted vocabulary.

Design specifications:
- Viridis colormap (default); two categories in viridis purple (#440154) and
  green (#32b67a)
- Font size 12 for ticks and legends, 14 for axis labels, colour-bar labels
  and titles, in every plot; nothing bold; legends without a frame, outside
  the plot
- By default each figure is sized around a fixed plot area (4.1 x 4.0 inches
  for the scatter, 4.0 x 3.9 for heatmaps, 8.7 x 3.9 for profiles), so long
  labels enlarge the figure instead of squeezing the plot, and colour bars run
  the plot's full height. plot_size=(width, height) chooses another area and
  scales text, markers and colour bars with it. figsize=(width, height) fixes
  the canvas instead, as before v0.8.2, and the plot takes what is left.

Since v0.8.2 importing this module no longer changes matplotlib's global
rcParams: every size is set on the artists it applies to.

Author: Amanda S. Barnard
Date: January 2026
"""

import math
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple, Union, Callable, Mapping, Sequence


# ============================================================================
# Sizes
# ============================================================================

# Point sizes at the default plot area. Through v0.8.1 these were written into
# plt.rcParams at import, and again inline on most labels, so a caller could
# not restyle the text even by setting rcParams first, and importing the module
# changed every other figure in the session. They now live here, and each plot
# applies them, scaled, to its own artists.
_BASE_FONT = {"tick": 12.0, "label": 14.0, "title": 14.0, "legend": 12.0,
              "annotation": 10.0}
_BASE_MARKER = 50.0          # scatter marker area, points^2
_BASE_SIZE_RANGE = (10.0, 200.0)

# The plot area (axes, in inches) each plot type receives at its default
# figsize, measured with matplotlib 3.10 on a nine-feature fit (the scatter
# with a colour bar). A plot drawn with plot_size equal to these numbers has
# exactly the default fonts and markers; any other plot_size scales the fonts
# by sqrt(area / reference area) and the marker areas by the area ratio.
_REFERENCE_PLOT_AREA = {
    "significance": (4.1, 4.0),
    "heatmap": (4.0, 3.9),
    "profile": (8.7, 3.9),
}

# Default characters kept of a sample label before the middle is elided.
DEFAULT_MAX_LABEL_LENGTH = 24


class _Style:
    """Font and marker sizes for one figure, scaled to its plot area."""

    def __init__(self, kind: str, plot_size: Optional[Tuple[float, float]]):
        if plot_size is None:
            self.scale = 1.0
        else:
            w, h = _validate_plot_size(plot_size)
            w0, h0 = _REFERENCE_PLOT_AREA[kind]
            self.scale = math.sqrt((w * h) / (w0 * h0))
        for role, size in _BASE_FONT.items():
            setattr(self, role, size * self.scale)
        self.marker = _BASE_MARKER * self.scale ** 2
        self.size_range = tuple(v * self.scale ** 2 for v in _BASE_SIZE_RANGE)


def _validate_plot_size(plot_size) -> Tuple[float, float]:
    try:
        w, h = (float(v) for v in plot_size)
    except (TypeError, ValueError):
        raise ValueError(f"plot_size must be (width, height) in inches, got {plot_size!r}")
    if not (w > 0 and h > 0 and math.isfinite(w) and math.isfinite(h)):
        raise ValueError(f"plot_size must be two positive numbers of inches, got {plot_size!r}")
    return w, h


def _default_plot_size(kind: str, figsize, plot_size, ax):
    """
    The plot area to lay the figure out around, or None for a fixed canvas.

    v0.8.2: with no figsize, plot_size or axes from the caller, the figure is
    built around the plot area the old default canvas gave, so fonts and
    markers are as before, but long labels grow the figure rather than squeeze
    the plot. Through v0.8.1 a heatmap with long sample labels kept its 6 x 5
    inch canvas and its cells shrank to a fraction of it.
    """
    if plot_size is None and figsize is None and ax is None:
        return _REFERENCE_PLOT_AREA[kind]
    return plot_size


def _new_axes(figsize, plot_size, ax, colorbar: bool = False):
    """
    The axes to draw on, and a colour-bar axes when the plot area is fixed.

    With plot_size the figure is created here and resized later by
    _fit_plot_area, so it cannot be combined with an axes the caller owns.
    """
    if plot_size is not None:
        if ax is not None:
            # only reachable with an explicit plot_size: the default is not
            # applied when the caller supplies ax
            raise ValueError(
                "plot_size sizes the whole figure around the plot, so it cannot be "
                "combined with ax. Pass one or the other."
            )
        w, h = _validate_plot_size(plot_size)
        # The axes start at their final size: seaborn decides how many tick
        # labels fit from the size the axes have when it draws.
        W, H = w + 2.5, h + 2.0
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([1.0 / W, 1.0 / H, w / W, h / H])
        cax = fig.add_axes([(1.15 + w) / W, 1.0 / H, 0.2 / W, h / H]) if colorbar else None
        return ax, cax
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax, None


def _fit_plot_area(ax, plot_size, cax=None, cbar_pad: float = 0.15,
                   cbar_width: float = 0.2, pad: float = 0.05) -> None:
    """
    Resize ax's figure so the plot area is exactly plot_size inches, with the
    margins set by what is drawn around it.

    Text is sized in points, so the margins it needs do not change when the
    figure grows; a few rounds of measure, resize and place therefore settle.
    Any colour bar sits beside the plot at the plot's full height.
    """
    fig = ax.figure
    w, h = _validate_plot_size(plot_size)
    margins = [1.0, 1.0, 1.0 + (cbar_pad + cbar_width if cax is not None else 0.0), 0.6]

    def place(m):
        left, bottom, right, top = m
        W, H = left + w + right, bottom + h + top
        fig.set_size_inches(W, H)
        ax.set_position([left / W, bottom / H, w / W, h / H])
        if cax is not None:
            cax.set_position([(left + w + cbar_pad) / W, bottom / H, cbar_width / W, h / H])

    for _ in range(6):
        place(margins)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tight = fig.get_tightbbox(renderer)
        box = ax.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
        new = [box.x0 - tight.x0 + pad, box.y0 - tight.y0 + pad,
               tight.x1 - box.x1 + pad, tight.y1 - box.y1 + pad]
        settled = max(abs(a - b) for a, b in zip(new, margins)) < 0.005
        margins = new
        if settled:
            break
    place(margins)


def _finish(ax, plot_size, cax=None) -> None:
    """Lay the figure out: around a fixed plot area, or with tight_layout."""
    if plot_size is not None:
        _fit_plot_area(ax, plot_size, cax=cax)
    else:
        # The axes' own figure. plt.tight_layout() acts on the current figure,
        # which is not the one drawn on when the caller passes ax from another.
        ax.figure.tight_layout()


def _label_capacity(length_in: float, fontsize: float) -> int:
    """
    How many text labels of this size fit side by side along length_in inches
    without their boxes touching. A label's box is about 1.15 times its font
    size across (ascent plus descent), whichever way it is rotated.
    """
    return max(1, int(length_in * 72.0 / (1.15 * fontsize)))


def _label_rotation(labels, slot_in: float, fontsize: float) -> int:
    """
    0 when every label fits upright in the space each one has, else 90.

    v0.8.2: feature labels were always turned on end, even two-character
    names with room to spare. Upright text is read without turning the head,
    so rotation is kept for labels that would otherwise run into each other.
    A character is taken as 0.6 of the font size across, which is about right
    for the default sans-serif, with a tenth of the slot left as a gap.
    """
    widest = max((len(str(t)) for t in labels), default=0) * 0.6 * fontsize
    return 0 if widest <= 0.9 * slot_in * 72.0 else 90


def _axes_inches(ax, colorbar_beside: bool = False) -> Tuple[float, float]:
    """
    The axes' size in inches as currently placed. A colour bar added later with
    colorbar(ax=ax) takes 20 per cent of the width (fraction 0.15, pad 0.05),
    so allow for it, with a little to spare.
    """
    box = ax.get_position()
    W, H = ax.figure.get_size_inches()
    width = box.width * W * (0.78 if colorbar_beside else 1.0)
    return width, box.height * H


def _elide(label, max_length: Optional[int]) -> str:
    """Shorten a label by removing its middle, so both ends stay readable."""
    text = str(label)
    if max_length is None or len(text) <= max_length:
        return text
    if max_length < 5:
        return text[:max_length]
    keep = max_length - 3
    head = (keep + 1) // 2
    return text[:head] + "..." + text[len(text) - (keep - head):]


def _display_name(name, capitalise: bool = False) -> str:
    """
    A column or category name as a label: underscores become spaces, and
    titles (legend and colour-bar) start with a capital. Sample identifiers are
    never passed through this, so an index can still be looked up as written.
    A first word that already has a capital in it (pH, mRNA) is left alone.
    """
    text = str(name).replace("_", " ")
    if capitalise and text.split(" ", 1)[0].islower():
        text = text[:1].upper() + text[1:]
    return text


def _sample_label(index_value, sample_labels, max_length) -> str:
    if sample_labels is None:
        label = index_value
    elif callable(sample_labels):
        label = sample_labels(index_value)
    else:
        label = sample_labels[index_value]
    return _elide(label, max_length)


def _feature_names(dt_fitted, plot_name: str, n_columns: Optional[int] = None) -> List:
    """
    Names of the features the explanations are over, for any fit type.

    v0.8.2: through v0.8.1 heatmap() and profile_plot() refused any fit without
    feature_columns_, which a text fit never sets, although the vocabulary was
    available the whole time. A tabular or graph fit names the columns it kept;
    a text fit names its terms. Every fit entry point clears both before it
    starts (v0.8.0), so a text fit after a tabular one cannot pick up the
    tabular names, and the count is checked against the explanations anyway.
    """
    names = None
    if getattr(dt_fitted, "feature_columns_", None) is not None \
            and getattr(dt_fitted, "keep_mask_", None) is not None:
        names = [dt_fitted.feature_columns_[i]
                 for i, keep in enumerate(dt_fitted.keep_mask_) if keep]
    elif getattr(dt_fitted, "vectorizer_", None) is not None:
        names = list(dt_fitted.vectorizer_.get_feature_names_out())
    if names is None:
        raise RuntimeError(
            f"{plot_name}() needs feature names, and this estimator has none: it "
            "has no fitted feature columns and no fitted vocabulary. Fit it on a "
            "DataFrame, an array, a graph or a list of documents first."
        )
    if n_columns is not None and len(names) != n_columns:
        raise RuntimeError(
            f"{plot_name}() found {len(names)} feature names for explanations over "
            f"{n_columns} features, so it cannot tell which column is which. The "
            "estimator's fitted state is inconsistent; refit it."
        )
    return names


def _normalised_feature_values(dt_fitted, sample_idx, names: List) -> np.ndarray:
    """
    The sample's value of each feature, scaled to [0, 1] across the training
    data, for colouring profile bars. NaN where the value was missing.
    """
    original = getattr(dt_fitted, "_df_original_fit", None)
    if original is not None and getattr(dt_fitted, "feature_columns_", None) is not None:
        data = original[names].to_numpy(dtype=np.float64)
        sample = original.loc[sample_idx, names].to_numpy(dtype=np.float64)
        # v0.8.2: nanmin and nanmax. With min and max, one missing value made
        # the whole column's range NaN, so that feature's bar was drawn
        # colourless for every sample, not only for the one that was missing
        # it, which is what the colourless bar is documented to mean.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo = np.nanmin(data, axis=0)
            hi = np.nanmax(data, axis=0)
        span = hi - lo
        span[~np.isfinite(span) | (span == 0)] = 1.0
        return (sample - lo) / span

    scaled = getattr(dt_fitted, "_text_scaled_fit_", None)
    if scaled is not None and getattr(dt_fitted, "train_index_", None) is not None:
        # A text fit's terms are already scaled to [0, 1] per column.
        row = scaled[dt_fitted.train_index_.get_loc(sample_idx)]
        return np.asarray(row.toarray(), dtype=np.float64).ravel()

    raise RuntimeError(
        "profile_plot() requires original tabular data, or a text fit's term "
        "matrix, to colour the bars by feature value; this estimator retains "
        "neither."
    )


def _cap_for_legibility(n_items: int, requested: Optional[int], capacity: int,
                        what: str, parameter: str) -> Optional[int]:
    """
    A default for an unset count when there are more items than fit legibly.

    v0.8.2: at 1,278 features a heatmap drew nothing but the separators between
    its columns, and a profile plot one black smear of overlapping tick labels,
    with no warning. A count the caller chose is always honoured.
    """
    if requested is not None or n_items <= capacity:
        return requested
    warnings.warn(
        f"{n_items} {what} are more than can be labelled legibly at this size, "
        f"so only the first {capacity} are shown. Pass {parameter}= to choose how "
        "many, or a larger plot_size or figsize to fit more.",
        UserWarning, stacklevel=3)
    return capacity


def _set_frame(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor('black')


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
        print(f"Warning: Column '{rank_column}' not found in results")
        return None

    # Check if column is all NaN (formative data not available)
    if results[rank_column].isna().all():
        print(f"Warning: Column '{rank_column}' has no data (likely fast_mode=True or selected_significance excludes this type)")

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
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = 'viridis',
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    plot_size: Optional[Tuple[float, float]] = None,
    **kwargs
) -> plt.Axes:
    """
    Create dual-perspective significance scatter plot (hero visualization).

    This plot reveals the relationship between actual significance (samples that
    ARE archetypal/prototypical/stereotypical) and formative significance
    (samples that CREATE the structure). It reads only the results frame, so
    it serves tabular, text and graph fits alike.

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
    figsize : tuple of float, optional
        A fixed canvas (width, height) in inches, the plot taking what the
        labels leave. By default the figure is sized around the plot area
        instead (see plot_size). Ignored when plot_size is given.
    cmap : str
        Colormap name (default: 'viridis')
    title : str, optional
        Plot title
    ax : plt.Axes, optional
        Existing axes to plot on
    plot_size : tuple of float, optional
        (width, height) of the plot area in inches. The figure is sized around
        it, and fonts, markers and the colour bar scale with it. Defaults to
        4.1 x 4.0, the area the old 6 x 5 canvas gave. New in v0.8.2.
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
    >>> # A 5 x 5 inch plot area for a publication figure
    >>> ax = significance_plot(results, 'archetypal', plot_size=(5, 5))
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

    plot_size = _default_plot_size("significance", figsize, plot_size, ax)
    style = _Style("significance", plot_size)

    # Check if formative data is available (could be None in fast_mode)
    if results[formative_col].isna().all():
        # Print informative message and skip plot
        print(f"\nWarning: Skipping significance plot:")
        print(f"  Formative data ('{formative_col}') not available")
        print(f"  Possible reasons: fast_mode=True, or selected_significance excludes '{significance}'")
        print(f"  To enable: DataTypical(shapley_mode=True, fast_mode=False)")

        # Return the caller's axes untouched, or a placeholder with a message
        if ax is None:
            ax, _ = _new_axes(figsize, plot_size, None)
            ax.text(0.5, 0.5, 'Formative data not available\n(fast_mode=True)',
                   ha='center', va='center', fontsize=style.label, color='gray')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel(f'{significance.capitalize()} Rank (Actual)', fontsize=style.label)
            ax.set_ylabel(f'{significance.capitalize()} Rank (Formative)', fontsize=style.label)
            ax.tick_params(labelsize=style.tick)
            if title:
                ax.set_title(title, fontsize=style.title)
            if plot_size is not None:
                _fit_plot_area(ax, plot_size)
            return ax
        return ax

    continuous = False
    if color_by is not None:
        if color_by not in results.columns:
            raise ValueError(f"color_by column '{color_by}' not found in results")
        c_values = results[color_by]
        is_numeric = pd.api.types.is_numeric_dtype(c_values)
        continuous = is_numeric and c_values.nunique() > 12

    ax, cax = _new_axes(figsize, plot_size, ax, colorbar=continuous)

    # Prepare size values (needed for all plot types)
    if size_by is not None:
        if size_by not in results.columns:
            raise ValueError(f"size_by column '{size_by}' not found in results")
        s_values = results[size_by]
        # Normalize to a range that scales with the plot area (10-200 at default)
        s_min, s_max = s_values.min(), s_values.max()
        lo, hi = style.size_range
        if s_max > s_min:
            s_normalized = lo + (hi - lo) * (s_values - s_min) / (s_max - s_min)
        else:
            s_normalized = style.marker
    else:
        s_normalized = style.marker

    # Handle coloring logic with automatic discrete/continuous detection
    if color_by is not None:
        n_unique = c_values.nunique()

        # Determine if discrete or continuous
        if not is_numeric:
            # Non-numeric -> discrete
            is_discrete = True
            use_markers = False
        elif n_unique < 6:
            # Numeric with <6 values -> discrete
            is_discrete = True
            use_markers = False
        elif 6 <= n_unique <= 12:
            # Numeric with 6-12 values -> discrete with markers
            is_discrete = True
            use_markers = True
        else:
            # Numeric with >12 values -> continuous
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
                # BINARY: purple (viridis[0.0]) for lower, green (viridis[0.66]) for higher
                viridis_cmap = matplotlib.colormaps.get_cmap('viridis')
                color_map = {
                    unique_values[0]: viridis_cmap(0.0),     # purple for lower value
                    unique_values[1]: viridis_cmap(0.66)     # green for higher value
                }
            else:
                # MULTI-CLASS: discrete viridis palette
                viridis_cmap = matplotlib.colormaps.get_cmap('viridis').resampled(n_categories)
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
                marker = marker_map[category] if use_markers else 'o'

                # Get size values for this subset
                if isinstance(s_normalized, (int, float)):
                    s_subset = s_normalized
                else:
                    s_subset = s_normalized[mask]

                ax.scatter(
                    results.loc[mask, actual_col],
                    results.loc[mask, formative_col],
                    c=[color_map[category]],
                    s=s_subset,
                    marker=marker,
                    edgecolors='black',
                    linewidth=0.5,
                    label=_display_name(category, capitalise=isinstance(category, str)),
                    **kwargs
                )

            # Legend outside the axes, with no frame
            ax.legend(
                title=_display_name(color_by, capitalise=True),
                fontsize=style.legend,
                title_fontsize=style.legend,
                loc='center left',
                bbox_to_anchor=(1.0, 0.5),
                frameon=False
            )

        else:
            # CONTINUOUS COLORING: Use colorbar
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
            if cax is not None:
                cbar = ax.figure.colorbar(scatter, cax=cax)
            else:
                cbar = ax.figure.colorbar(scatter, ax=ax)
            cbar.set_label(_display_name(color_by, capitalise=True), fontsize=style.label)
            cbar.ax.tick_params(labelsize=style.tick)

    else:
        # NO COLORING: one viridis colour (v0.8.2; was steelblue, outside viridis)
        ax.scatter(
            results[actual_col],
            results[formative_col],
            c=[matplotlib.colormaps.get_cmap('viridis')(0.33)],
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
                fontsize=style.annotation,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='gray', lw=0.5)
            )

    # Labels and title
    ax.set_xlabel(f'{significance.capitalize()} Rank (Actual)', fontsize=style.label)
    ax.set_ylabel(f'{significance.capitalize()} Rank (Formative)', fontsize=style.label)
    ax.tick_params(labelsize=style.tick)

    if title:
        ax.set_title(title, fontsize=style.title)
    else:
        ax.set_title(f'{significance.capitalize()} Dual-Perspective Analysis', fontsize=style.title)

    # Grid for readability
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Ensure limits include full range
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    _finish(ax, plot_size, cax)

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
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = 'viridis',
    center: Optional[float] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    plot_size: Optional[Tuple[float, float]] = None,
    sample_labels: Optional[Union[Mapping, pd.Series, Callable]] = None,
    max_label_length: Optional[int] = DEFAULT_MAX_LABEL_LENGTH,
    **kwargs
) -> plt.Axes:
    """
    Create feature attribution heatmap for Shapley explanations.

    Always shows explanations (why samples ARE significant). Features are always
    ordered by global importance (average across all samples). Instances can be
    ordered by actual or formative ranks. Serves tabular, text and graph fits:
    a text fit's features are its vocabulary terms.

    Parameters
    ----------
    dt_fitted : DataTypical
        Fitted DataTypical instance with shapley_mode=True
    results : pd.DataFrame, optional
        Results dataframe from fit_transform(). Required if samples not specified.
    samples : list or array, optional
        Positions of samples to show. If None, uses top_n samples from results.
    significance : str
        Which significance to visualize: 'archetypal', 'prototypical', 'stereotypical'
    order : str
        How to order instances (rows): 'actual' or 'formative'
        'actual': Order by {significance}_rank (samples that ARE significant)
        'formative': Order by {significance}_shapley_rank (samples that CREATE structure)
    top_n : int, optional
        Number of top samples to show (if samples not specified).
        Defaults to shapley_top_n from fitted model, or 20 if not set, and to
        no more rows than can be labelled legibly (with a warning).
    top_features : int, optional
        Number of top features to show (by global importance). Defaults to all
        of them, or to as many as can be labelled legibly when there are more
        (with a warning), so a wide vocabulary does not render as a blank box.
    figsize : tuple of float, optional
        A fixed canvas (width, height) in inches, the plot taking what the
        labels leave. By default the figure is sized around the plot area
        instead (see plot_size). Ignored when plot_size is given.
    cmap : str
        Colormap name (default: 'viridis')
    center : float, optional
        Value to center colormap at (for diverging colormaps)
    title : str, optional
        Plot title
    ax : plt.Axes, optional
        Existing axes to plot on
    plot_size : tuple of float, optional
        (width, height) of the heatmap area in inches. The figure is sized
        around it, fonts scale with it, and the colour bar runs its full height.
        Defaults to 4.0 x 3.9, the area the old 6 x 5 canvas gave. New in v0.8.2.
    sample_labels : mapping, Series or callable, optional
        Row labels: looked up by sample index, or computed by calling with it.
        Defaults to the index itself. New in v0.8.2.
    max_label_length : int or None
        Labels longer than this keep their start and end and lose the middle.
        None keeps them whole. New in v0.8.2.
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

    plot_size = _default_plot_size("heatmap", figsize, plot_size, ax)
    style = _Style("heatmap", plot_size)
    top_n_given = top_n is not None

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
        print(f"\nWarning: Skipping {significance} explanations heatmap:")
        print(f"  Explanations data not available for '{significance}'")
        print(f"  Possible reasons: shapley_mode=False, or selected_significance excludes '{significance}'")
        if significance == 'stereotypical':
            print(f"  Note: Stereotypical also requires stereotype_column to be set")

        print(f"\n  To enable this plot, refit with:")
        if significance == 'stereotypical':
            print(f"    DataTypical(shapley_mode=True, stereotype_column='<column>')")
        else:
            print(f"    DataTypical(shapley_mode=True)  # or set selected_significance='{significance}'")

        # Return empty axes with message
        ax, _ = _new_axes(figsize, plot_size, ax)
        ax.text(0.5, 0.5, f'{significance.capitalize()} explanations\nnot available',
               ha='center', va='center', fontsize=style.label, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        if plot_size is not None:
            _fit_plot_area(ax, plot_size)
        
        return ax

    feature_names = _feature_names(dt_fitted, "heatmap", Phi.shape[1])
    with_cbar = kwargs.get('cbar', True)
    ax, cax = _new_axes(figsize, plot_size, ax, colorbar=with_cbar)
    width_in, height_in = (plot_size if plot_size is not None
                           else _axes_inches(ax, colorbar_beside=bool(with_cbar)))

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
                print(f"\nWarning: order='formative' requested but formative data not available")
                print(f"  Falling back to order='actual'")
                rank_col = f"{significance}_rank"
                order = 'actual'
        else:
            raise ValueError(f"order must be 'actual' or 'formative', got '{order}'")

        if rank_col not in results.columns:
            raise RuntimeError(f"Cannot find {rank_col} in results")

        if not top_n_given:
            top_n = _cap_for_legibility(
                min(top_n, len(results)), None,
                _label_capacity(height_in, style.tick), "samples", "top_n") or top_n

        # Get DataFrame indices of top samples (already ordered by rank)
        top_samples_df_indices = results.nlargest(top_n, rank_col).index

        # Use get_shapley_explanations to handle shapley_top_n subsampling
        Phi_subset_list = []
        row_labels = []

        for df_idx in top_samples_df_indices:
            try:
                # Get explanations using the API (handles index mapping internally)
                explanations = dt_fitted.get_shapley_explanations(df_idx)
                shapley_values = explanations[significance]

                # Only include if we have valid data
                if np.any(shapley_values != 0) or not hasattr(dt_fitted, 'shapley_top_n') or dt_fitted.shapley_top_n is None:
                    Phi_subset_list.append(shapley_values)
                    row_labels.append(_sample_label(df_idx, sample_labels, max_label_length))
            except (IndexError, KeyError):
                # This sample doesn't have explanations computed
                pass

        if len(Phi_subset_list) == 0:
            print(f"\nError: None of the top {top_n} {significance} instances have explanations")
            print(f"  This can happen when shapley_top_n is too small")
            if cax is not None:
                cax.remove()
            ax.text(0.5, 0.5, f'No explanations available\nfor top {top_n} instances',
                   ha='center', va='center', fontsize=style.label, color='gray')
            ax.axis('off')
            if plot_size is not None:
                _fit_plot_area(ax, plot_size)
            return ax

        Phi_subset = np.array(Phi_subset_list)

        # WARNING: Check for zero Shapley values when using formative ordering
        if order == 'formative':
            zero_count = (Phi_subset == 0).all(axis=1).sum()
            if zero_count > 0:
                print(f"\nWarning: {zero_count}/{len(row_labels)} top formative {significance} instances have zero Shapley values")
                print(f"  This occurs when a formative instance is not in the top {significance} instances")
                print(f"  (determined by shapley_top_n parameter)")
                print(f"  These instances CREATE structure but are not themselves highly {significance}")

    else:
        # Custom sample list provided: positions, kept in the order given
        samples = np.asarray(samples)
        Phi_subset = Phi[samples, :]
        if dt_fitted.train_index_ is not None:
            row_labels = [_sample_label(dt_fitted.train_index_[s], sample_labels, max_label_length)
                          for s in samples]
        else:
            row_labels = [f"Sample {s}" for s in samples]

    # ALWAYS order features by global importance (average across ALL samples)
    global_importance = np.abs(Phi).mean(axis=0)
    feature_order = np.argsort(global_importance)[::-1]
    Phi_subset = Phi_subset[:, feature_order]
    feature_names = [feature_names[i] for i in feature_order]

    # Select top features: as asked, or as many as fit legibly
    top_features = _cap_for_legibility(
        len(feature_names), top_features,
        _label_capacity(width_in, style.tick), "features", "top_features")
    if top_features is not None:
        if top_features < 1:
            raise ValueError(f"top_features must be >= 1, got {top_features}")
        Phi_subset = Phi_subset[:, :top_features]
        feature_names = feature_names[:top_features]

    df_heatmap = pd.DataFrame(Phi_subset, index=row_labels,
                              columns=[_display_name(n) for n in feature_names])

    heatmap_kwargs = {
        'cmap': cmap,
        'center': center,
        'cbar_kws': {'label': 'Shapley Value'} if cax is not None
                    else {'label': 'Shapley Value', 'shrink': 0.5},
        # Separators only while the cells are wide enough to show colour
        # between them; at hundreds of columns they covered the whole plot.
        'linewidths': 0.5 if len(feature_names) <= 60 else 0.0,
        # Every row and column labelled when the labels fit without touching;
        # otherwise seaborn labels a regular subset rather than overprinting.
        'xticklabels': True if len(feature_names) <= _label_capacity(width_in, style.tick) else 'auto',
        'yticklabels': True if len(row_labels) <= _label_capacity(height_in, style.tick) else 'auto',
        'annot': False,
        'fmt': '.3f',
        'square': False
    }
    if cax is not None:
        heatmap_kwargs['cbar_ax'] = cax
    heatmap_kwargs.update(kwargs)

    # Seaborn thins crowded tick labels using the size they have when it draws,
    # so they get their final size first.
    ax.tick_params(labelsize=style.tick)
    sns.heatmap(df_heatmap, ax=ax, **heatmap_kwargs)

    _set_frame(ax)
    if ax.collections and ax.collections[0].colorbar is not None:
        cbar = ax.collections[0].colorbar
        cbar.outline.set_linewidth(1)
        cbar.outline.set_edgecolor('black')
        cbar.ax.yaxis.label.set_size(style.label)
        cbar.ax.tick_params(labelsize=style.tick)

    # Labels: the same sizes as the other two plots (v0.8.2; these were 12)
    ax.set_xlabel('Features (Ordered by Global Importance)', fontsize=style.label)
    rank_kind = 'Actual' if order == 'actual' else 'Formative'
    ax.set_ylabel(f'Samples (by {rank_kind} Rank)', fontsize=style.label)
    tick_names = [t.get_text() for t in ax.get_xticklabels()]
    ax.tick_params(axis='x', labelsize=style.tick, labelrotation=_label_rotation(
        tick_names, width_in / max(len(tick_names), 1), style.tick))
    ax.tick_params(axis='y', labelrotation=0, labelsize=style.tick)

    if title:
        ax.set_title(title, fontsize=style.title)
    else:
        ax.set_title(f'{significance.title()} Explanations', fontsize=style.title)

    _finish(ax, plot_size, cax)

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
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = 'viridis',
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    plot_size: Optional[Tuple[float, float]] = None,
    max_label_length: Optional[int] = DEFAULT_MAX_LABEL_LENGTH,
    **kwargs
) -> plt.Axes:
    """
    Create feature importance profile for a single sample.

    Shows Shapley explanations (y-axis) for each feature (x-axis), with bars colored by
    the normalized feature value for this sample. Features can be ordered locally (by
    this sample's importance) or globally (by average importance across all samples).
    Serves tabular, text and graph fits; a text fit's bars are its terms, coloured
    by the document's scaled term weight.

    Missingness Indicator: a feature this sample was missing in the original data
    appears as a colourless (transparent) bar, while observed features are
    coloured. This preserves the signal that imputation was used and
    distinguishes real from imputed data. Bar height (Shapley value) is always
    shown regardless of missingness.

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
        Number of top features to display. Defaults to all of them, or to as
        many as can be labelled legibly when there are more (with a warning).
        Features are selected after ordering (top N by importance).
    figsize : tuple of float, optional
        A fixed canvas (width, height) in inches, the plot taking what the
        labels leave. By default the figure is sized around the plot area
        instead (see plot_size). Ignored when plot_size is given.
    cmap : str
        Colormap name (default: 'viridis')
    title : str, optional
        Plot title
    ax : plt.Axes, optional
        Existing axes to plot on
    plot_size : tuple of float, optional
        (width, height) of the plot area in inches. The figure is sized around
        it, fonts scale with it, and the colour bar runs its full height.
        Defaults to 8.7 x 3.9, the area the old 12 x 5 canvas gave. New in
        v0.8.2.
    max_label_length : int or None
        The sample's label in the default title keeps its start and end and
        loses the middle beyond this length. None keeps it whole. New in v0.8.2.
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

    if top_features is not None and top_features < 1:
        raise ValueError(f"top_features must be >= 1, got {top_features}")

    plot_size = _default_plot_size("profile", figsize, plot_size, ax)
    style = _Style("profile", plot_size)

    # Get explanations for this sample
    explanations = dt_fitted.get_shapley_explanations(sample_idx)
    if significance not in explanations:
        raise RuntimeError(
            f"No explanations available for '{significance}'. "
            f"Refit with DataTypical(shapley_mode=True) and ensure "
            f"selected_significance is None or includes '{significance}'."
        )
    shapley_values = explanations[significance]

    feature_names = _feature_names(dt_fitted, "profile_plot", shapley_values.shape[0])

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
        if Phi_explanations is None:
            raise RuntimeError(
                f"Global ordering requires {significance} explanations, which are not available. "
                f"Use order='local', or refit ensuring selected_significance includes '{significance}'."
            )
        importance = np.mean(np.abs(Phi_explanations), axis=0)
        ordering_type = "global"

    # Sort features by importance (descending by absolute value)
    sorted_idx = np.argsort(importance)[::-1]

    # Get sorted Shapley values (keep original signs for plotting)
    shapley_sorted = shapley_values[sorted_idx]
    features_sorted = [feature_names[i] for i in sorted_idx]

    # The sample's own feature values, scaled to [0, 1], colour the bars
    normalized_values = _normalised_feature_values(dt_fitted, sample_idx, feature_names)
    normalized_sorted = normalized_values[sorted_idx]

    ax, cax = _new_axes(figsize, plot_size, ax, colorbar=True)
    width_in = (plot_size[0] if plot_size is not None
                else _axes_inches(ax, colorbar_beside=True)[0])

    # Apply top_features: as asked, or as many as fit legibly
    shown = _cap_for_legibility(len(features_sorted), top_features,
                                _label_capacity(width_in, style.tick),
                                "features", "top_features")
    if shown is not None and shown < len(features_sorted):
        shapley_sorted = shapley_sorted[:shown]
        features_sorted = features_sorted[:shown]
        normalized_sorted = normalized_sorted[:shown]

    # Create color array from normalized feature values (NaN draws transparent)
    colors = matplotlib.colormaps.get_cmap(cmap)(normalized_sorted)

    # Create bar plot - use signed Shapley values (can be negative)
    x_pos = np.arange(len(features_sorted))
    ax.bar(x_pos, shapley_sorted, color=colors, edgecolor='black', linewidth=0.5, **kwargs)

    # Add zero reference line
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    if cax is not None:
        cbar = ax.figure.colorbar(sm, cax=cax)
    else:
        cbar = ax.figure.colorbar(sm, ax=ax)
    cbar.set_label('Normalized Feature Value', fontsize=style.label)
    cbar.ax.tick_params(labelsize=style.tick)

    # Labels: all of them when they fit; past that (only when the caller asked
    # for more), every k-th, so that no two labels print over each other
    every = max(1, math.ceil(len(features_sorted) / _label_capacity(width_in, style.tick)))
    tick_names = [_display_name(n) for n in features_sorted[::every]]
    rotation = _label_rotation(tick_names, width_in * every / max(len(features_sorted), 1),
                               style.tick)
    ax.set_xticks(x_pos[::every])
    ax.set_xticklabels(tick_names, rotation=rotation, ha='center', fontsize=style.tick)
    ax.tick_params(axis='y', labelsize=style.tick)

    # Update xlabel based on ordering type and how many features are shown
    if shown is not None and shown < len(feature_names):
        feature_label = f'Top {shown} Features (Ordered by {ordering_type.capitalize()} Importance)'
    else:
        feature_label = f'Features (Ordered by {ordering_type.capitalize()} Importance)'

    ax.set_xlabel(feature_label, fontsize=style.label)
    ax.set_ylabel('Shapley Value', fontsize=style.label)

    if title:
        ax.set_title(title, fontsize=style.title)
    else:
        sig_name = significance.capitalize()
        ax.set_title(f'{sig_name} Explanations: {_elide(sample_idx, max_label_length)}',
                     fontsize=style.title)

    # Grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    _finish(ax, plot_size, cax)

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
