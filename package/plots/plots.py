from typing import Optional
from typing import Sequence, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
from odsmr.constants import STATE_BOUNDS


def plot_trajectory_seaborn(trajectory, maintenance, filename=None, width=14, height=7, window=10):
    """
    Plots degradation trajectories with local ±window maintenance highlights.
    The last column of `trajectory` must be a boolean maintenance array.
    """

    sns.set_theme(style="darkgrid", context="paper")

    indicators = list(STATE_BOUNDS.keys())

    # Separate indicators and maintenance flag
    maintenance = maintenance.astype(bool)
    Y = trajectory

    T = len(Y)
    x = np.arange(T)

    fig, ax = plt.subplots(figsize=(width, height))
    palette = sns.color_palette("tab10", Y.shape[1])

    # Plot trajectories
    for i in range(Y.shape[1]):
        ax.plot(
            x,
            Y[:, i],
            label=indicators[i],
            color=palette[i],
            linewidth=2,
            alpha=0.9
        )

    # --- Highlight ±window around each maintenance event --------------------
    highlight_color = "#DDAA77"  # soft brown/orange like in your sample
    highlight_alpha = 0.25

    maintenance_indices = np.where(maintenance)[0]
    added_legend = False

    for t in maintenance_indices:
        start = max(0, t - window)
        end   = min(T - 1, t + window)
        ax.axvspan(
            start, end,
            color=highlight_color,
            alpha=highlight_alpha,
            zorder=0,
            label="Maintenance (±10 steps)" if not added_legend else None
        )
        added_legend = True

    # Labels & Title
    # ax.set_title("Degradation Trajectory", fontsize=18, weight="bold")
    ax.set_xlabel("Time Step", fontsize=14)
    ax.set_ylabel("Indicator Value", fontsize=14)

    ax.set_xlim(0, len(trajectory))

    # Legend (both health indicators + maintenance)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        title="Health Indicators",
        loc="lower left",
        frameon=True,
        fontsize=10,
        title_fontsize=12
    )

    sns.despine()
    plt.tight_layout()
    if filename is not None:
        plt.savefig("trajectory.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()
    
    
def plot_measures_seaborn(sequence, input_feautres, filename=None):
    """
    Seaborn-based visualization of 7 sensor measurements
    with only one x-axis label per row and one y-axis label per column.
    All ticks are preserved.
    """
    sns.set_theme(style="darkgrid", context="paper")  
    n_sensors = len(input_feautres)

    fig, axs = plt.subplots(4, 7, figsize=(14, 7))
    axs = axs.flatten()

    palette = sns.color_palette("Dark2", n_sensors)

    for sensor_id in range(n_sensors):
        ax = axs[sensor_id]
        label = input_feautres[sensor_id]

        ax.plot(
            sequence[:, sensor_id],
            color=palette[sensor_id],
            linewidth=1.4,
            alpha=0.95
        )

        ax.set_title(label, fontsize=12, weight="bold")
        ax.grid(True, alpha=0.3)

        row = sensor_id // 4   # 0 or 1
        col = sensor_id % 7    # 0,1,2,3

        # ---- Y-axis labels: only at the leftmost column ----
        if col == 0:
            ax.set_ylabel("Value", fontsize=10)
        else:
            ax.set_ylabel(None)         # hide label
            # KEEP TICKS → do NOT remove them
            ax.tick_params(axis="y", labelleft=True)

        # ---- X-axis labels: only at the bottom row ----
        if row == 1:
            ax.set_xlabel("Time Step", fontsize=10)
        else:
            ax.set_xlabel(None)         # hide label
            # KEEP TICKS → do NOT remove them
            ax.tick_params(axis="x", labelbottom=True)

    # Hide the unused 8th panel
    if n_sensors < len(axs):
        for k in range(n_sensors, len(axs)):
            axs[k].set_visible(False)

    plt.tight_layout(pad=0.2)
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.show()
    
def plot_measures_distribution(df, variable_order, phase_order, save_path: Optional[str]=None):
    sns.set_theme(style="ticks", palette="pastel")
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    sns.boxplot(
        data=df,
        x="variable",
        y="value",
        hue="phase",
        order=variable_order,
        hue_order=phase_order,
        dodge=True,
        ax=ax,
        showfliers=False  # optional: hide outliers for cleaner look
    )

    ax.set_xlabel("Input variables")
    ax.set_ylabel("Scaled values")
    ax.tick_params(axis='x', rotation=25)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Phase", loc="upper right")#bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    
def plot_indicators_distribution(df, output_features, save_path: Optional[str]=None):
    sns.set_theme(style="ticks", palette="pastel")
    fig, axs = plt.subplots(1,1, figsize=(10,4))
    # sns.boxplot(outputs_scaled)
    sns.boxplot(df[output_features])#, cut=0)
    axs.set_xticks(range(10))
    axs.set_xticklabels(output_features, rotation=25, ha="right")
    plt.xlabel("Output variables")
    plt.ylabel("Scaled values")
    plt.grid()
    if save_path is not None:
        plt.savefig("outputs_distribution_bx.pdf", bbox_inches='tight')
    plt.show()
    
def plot_obs_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    variable_names=None,
    layout: str = "2x5",
    x_label: str = "Time",
    y_label: str = "Value",
    palette=("C0", "C1"),      # Seaborn/Matplotlib named colors
    linewidth: float = 1.8,
    style: str = "whitegrid",  # seaborn style: "white", "whitegrid", "ticks", "darkgrid"
    height_per_row: float = 2.6,
    width_per_col: float = 3.8,
    legend_loc: str = "bottom",  # "bottom" | "top" | "right"
    show: bool = True,
    savepath: Optional[str]=None
):
    """
    Simple Seaborn-based multi-plot for (T, 10) observations vs predictions.
    - 10 subplots, one per variable
    - Single figure-level x/y labels
    - One global legend placed outside to avoid overlaps
    """
    # --- Validate ---
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("y_true and y_pred must be 2D arrays (T, 10).")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    T, D = y_true.shape
    if D != 10:
        raise ValueError(f"Expected second dimension = 10, got {D}.")

    # Layout parsing
    r_str, c_str = layout.lower().split("x")
    n_rows, n_cols = int(r_str), int(c_str)
    if n_rows * n_cols != D:
        raise ValueError(f"layout {layout} must satisfy rows*cols == 10")

    # Variable names
    if variable_names is None:
        variable_names = [f"Var {i+1}" for i in range(D)]
    elif len(variable_names) != D:
        raise ValueError("variable_names must have length 10")

    # Seaborn theme
    sns.set_theme(style=style, context="notebook")

    # Figure size (a little extra height to avoid crowding)
    figsize = (max(10, width_per_col * n_cols), max(4.5, height_per_row * n_rows) + 0.4)

    # Constrained layout + tiny bottom/left margins for sup labels
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(n_rows, n_cols)

    x = np.arange(T)
    for idx in range(D):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]

        # Seaborn lines
        sns.lineplot(x=x, y=y_true[:, idx], ax=ax, color=palette[0], linewidth=linewidth, label="Observation")
        sns.lineplot(x=x, y=y_pred[:, idx], ax=ax, color=palette[1], linewidth=linewidth, label="Prediction", alpha=0.95)

        # Title with a bit of padding (prevents touching legend/title region)
        ax.set_title(str(variable_names[idx]), fontsize=11, pad=8)

        # Grid (major + minor)
        ax.minorticks_on()
        ax.grid(True, which="major", linestyle=":", alpha=0.6)
        ax.grid(True, which="minor", linestyle=":", alpha=0.25)

        # Only show bottom row x tick labels / left column y tick labels (cleaner)
        if r != n_rows - 1:
            ax.tick_params(labelbottom=False)
        if c != 0:
            ax.tick_params(labelleft=False)
        
        if ax.legend_:
            ax.legend_.remove()

    # Build a single legend from the first axes
    handles, labels = axes[0, 0].get_legend_handles_labels()

    if legend_loc == "bottom":
        fig.legend(handles, labels,
                   loc="center",
                   bbox_to_anchor=(0.9, -0.035),  # Bottom outside
                   bbox_transform=fig.transFigure,
                   ncol=2, frameon=False)
    elif legend_loc == "top":
        fig.legend(handles, labels,
                   loc="center",
                   bbox_to_anchor=(0.5, 1.0),  # Top outside
                   bbox_transform=fig.transFigure,
                   ncol=2, frameon=False)
    else:  # right
        fig.legend(handles, labels,
                   loc="center left",
                   bbox_to_anchor=(1.0, 0.5),
                   bbox_transform=fig.transFigure,
                   ncol=1, frameon=False)

    # Single global labels (slightly inset to avoid tick/label collisions)
    fig.supxlabel(x_label, y=-0.035)  # a bit above bottom edge
    fig.supylabel(y_label, x=-0.01)  # a bit to the right of left edge

    # If you still see crowding with very long titles, you can bump figure height:
    # fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1] + 0.3)

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, axes

def plot_cv_losses(training_losses, validation_losses):
    """
    training_losses: list of lists (n_folds x n_epochs)
    validation_losses: list of lists (n_folds x n_epochs)
    """

    train_losses = np.array(training_losses)
    val_losses = np.array(validation_losses)

    train_mean = train_losses.mean(axis=0)
    train_std = train_losses.std(axis=0)

    val_mean = val_losses.mean(axis=0)
    val_std = val_losses.std(axis=0)

    epochs = np.arange(1, len(train_mean) + 1)

    plt.figure(figsize=(8, 5))

    # Train
    plt.plot(epochs, train_mean, label="Train Loss (Mean)")
    plt.fill_between(
        epochs,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.3
    )

    # Validation
    plt.plot(epochs, val_mean, label="Validation Loss (Mean)")
    plt.fill_between(
        epochs,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.3
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("5-Fold Cross Validation (Mean ± Std)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

def plot_multiple_obs_vs_pred(
    y_true: np.ndarray,                         # (T, D)
    y_pred: np.ndarray,                         # (T, D) OR (n_models, T, D)
    variable_names: Optional[Sequence[str]] = None,
    model_names: Optional[Sequence[str]] = None,
    variables: Optional[Sequence[Union[int, str]]] = None,  # indices or names to plot
    layout: str = "auto",                       # "auto" or like "2x3"
    x_label: str = "Time",
    y_label: str = "Value",

    # --- Color & style controls ---
    palette: Union[str, Sequence] = "colorblind",   # good default for distinguishability
    linewidth: float = 1.8,                         # base linewidth for models
    alpha: float = 0.95,                             # kept for backward compat (not used if model_alpha given)
    model_alpha: Optional[Sequence[float]] = None,   # per-model alpha, e.g. [0.9, 0.7, 0.5]
    model_linestyles: Optional[Sequence[str]] = None,# e.g. ["solid", "dashed", "dashdot", (0, (1,1))]
    model_markers: Optional[Sequence[Optional[str]]] = None, # e.g. ["o", "s", "D", None]
    marker_every: Optional[int] = None,              # e.g. 10 (show marker every N points)
    obs_color: Optional[str] = "black",
    obs_linewidth: float = 2.3,
    obs_style: str = "solid",                        # style for observation line
    obs_alpha: float = 1.0,

    style: str = "whitegrid",                   # seaborn style: "white", "whitegrid", "ticks", "darkgrid"
    height_per_row: float = 2.6,
    width_per_col: float = 3.8,
    legend_loc: str = "bottom",                 # "bottom" | "top" | "right"
    legend_ncol: Optional[int] = None,          # columns in legend
    show: bool = True,
    savepath: Optional[str] = None,

    # Optional: emphasize one model by index
    highlight_model: Optional[int] = None,      # e.g., 0 to highlight first model
    highlight_linewidth: float = 2.6,
    highlight_alpha: float = 1.0,
):
    """
    Seaborn-based multi-plot for observations vs predictions with clarity-focused styling.

    Key features:
    - Plot a subset of variables with `variables` (indices or names).
    - Multi-model support: y_pred can be (T, D) or (n_models, T, D).
    - Distinguish models using alpha, line style, markers, and colorblind-friendly palette.
    - Global legend outside the axes; observation line is thicker and on top.

    Returns
    -------
    fig, axes
    """

    # --- Validate y_true ---
    if y_true.ndim != 2:
        raise ValueError("y_true must be a 2D array of shape (T, D).")
    T, D = y_true.shape

    # --- Normalize y_pred to (n_models, T, D) ---
    if y_pred.ndim == 2:
        if y_pred.shape != (T, D):
            raise ValueError("When y_pred is 2D, it must match y_true shape (T, D).")
        y_pred_models = y_pred[None, ...]  # (1, T, D)
    elif y_pred.ndim == 3:
        if y_pred.shape[1:] != (T, D):
            raise ValueError("When y_pred is 3D, it must be (n_models, T, D) matching y_true.")
        y_pred_models = y_pred
    else:
        raise ValueError("y_pred must be either (T, D) or (n_models, T, D).")
    n_models = y_pred_models.shape[0]

    # --- Variable names ---
    if variable_names is None:
        variable_names = [f"Var {i+1}" for i in range(D)]
    elif len(variable_names) != D:
        raise ValueError("variable_names must have length equal to D.")

    # --- Select subset of variables ---
    if variables is None:
        idx_list = list(range(D))
    else:
        idx_list = []
        for v in variables:
            if isinstance(v, int):
                if not (0 <= v < D):
                    raise ValueError(f"Variable index {v} out of range [0, {D-1}].")
                idx_list.append(v)
            elif isinstance(v, str):
                try:
                    idx_list.append(variable_names.index(v))
                except ValueError:
                    raise ValueError(f"Variable name '{v}' not found in variable_names.")
            else:
                raise TypeError("Each entry in `variables` must be int or str.")
        # deduplicate preserving order
        seen = set()
        idx_list = [i for i in idx_list if not (i in seen or seen.add(i))]

    K = len(idx_list)
    if K == 0:
        raise ValueError("No variables selected to plot.")

    # Slice data to selected variables
    y_true_sel = y_true[:, idx_list]                 # (T, K)
    y_pred_sel = y_pred_models[:, :, idx_list]       # (n_models, T, K)
    variable_names_sel = [variable_names[i] for i in idx_list]

    # --- Model names ---
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(n_models)]
    else:
        if len(model_names) != n_models:
            raise ValueError("model_names length must match n_models.")

    # --- Theme ---
    sns.set_theme(style=style, context="notebook")

    # --- Colors: 1 for obs + 1 per model ---
    def _resolve_palette(pal, n_needed):
        if isinstance(pal, str):
            return list(sns.color_palette(pal, n_needed))
        cols = list(pal)
        if len(cols) < n_needed:
            cols.extend(sns.color_palette("tab10", n_needed - len(cols)))
        return cols[:n_needed]

    colors = _resolve_palette(palette, n_models + 1)  # [obs_color, m1, m2, ...]
    if obs_color is not None:
        colors[0] = obs_color

    # --- Alpha (transparency) handling ---
    if model_alpha is None:
        # default: fade later models a bit more
        # e.g., for 3 models -> [0.95, 0.75, 0.6]
        if n_models == 1:
            alpha_list = [0.9]
        else:
            base = 0.95
            step = 0.25 / max(1, n_models - 1)
            alpha_list = [max(0.4, base - i * step) for i in range(n_models)]
    else:
        if len(model_alpha) != n_models:
            raise ValueError("model_alpha length must match n_models.")
        alpha_list = list(model_alpha)

    # --- Line styles (distinct patterns help a lot) ---
    if model_linestyles is None:
        # Common distinct styles; repeat if needed
        base_styles = ["solid", "dashed", "dashdot", (0, (1, 1)), (0, (3, 1, 1, 1))]
        model_linestyles = [base_styles[i % len(base_styles)] for i in range(n_models)]
    else:
        if len(model_linestyles) != n_models:
            raise ValueError("model_linestyles length must match n_models.")

    # --- Markers (optional) ---
    if model_markers is not None and len(model_markers) != n_models:
        raise ValueError("model_markers length must match n_models when provided.")

    # --- Layout ---
    if layout == "auto":
        cols = int(np.ceil(np.sqrt(K)))
        rows = int(np.ceil(K / cols))
        n_rows, n_cols = rows, cols
    else:
        try:
            r_str, c_str = layout.lower().split("x")
            n_rows, n_cols = int(r_str), int(c_str)
        except Exception:
            raise ValueError("layout must be 'auto' or a string like '2x3'.")
        if n_rows * n_cols != K:
            raise ValueError(f"layout {layout} must satisfy rows*cols == {K}.")

    # --- Figure ---
    figsize = (
        max(6.0, width_per_col * n_cols),
        max(3.6, height_per_row * n_rows) + 0.4,
    )
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(n_rows, n_cols)

    x = np.arange(T)

    # --- Plot ---
    for plot_idx in range(K):
        r, c = divmod(plot_idx, n_cols)
        ax = axes[r, c]

        # 1) Models first (lower zorder), so observation stays clearly on top
        for m in range(n_models):
            lw = highlight_linewidth if (highlight_model is not None and m == highlight_model) else linewidth
            a = highlight_alpha if (highlight_model is not None and m == highlight_model) else alpha_list[m]
            ls = model_linestyles[m]
            mk = None if model_markers is None else model_markers[m]

            sns.lineplot(
                x=x,
                y=y_pred_sel[m, :, plot_idx],
                ax=ax,
                color=colors[m + 1],
                linewidth=lw,
                alpha=a,
                linestyle=ls,
                marker=mk,
                markevery=marker_every if mk is not None else None,
                label=model_names[m],
                zorder=2,  # below observation
            )

        # 2) Observation last (higher zorder, thicker)
        sns.lineplot(
            x=x,
            y=y_true_sel[:, plot_idx],
            ax=ax,
            color=colors[0],
            linewidth=obs_linewidth,
            alpha=obs_alpha,
            linestyle=obs_style,
            label="Observation",
            zorder=3,
        )

        # Title & grid
        ax.set_title(str(variable_names_sel[plot_idx]), fontsize=14, pad=8)
        ax.minorticks_on()
        ax.grid(True, which="major", linestyle=":", alpha=0.6)
        ax.grid(True, which="minor", linestyle=":", alpha=0.25)

        # Clean ticks
        if r != n_rows - 1:
            ax.tick_params(labelbottom=False)
        if c != 0:
            ax.tick_params(labelleft=False)

        # Remove local legends
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    # Hide unused cells (when layout auto makes extra)
    total_cells = n_rows * n_cols
    if total_cells > K:
        for extra_idx in range(K, total_cells):
            r, c = divmod(extra_idx, n_cols)
            axes[r, c].axis("off")

    # --- Legend (build from first data axes) ---
    first_ax = None
    for ax in axes.flat:
        if ax.has_data():
            first_ax = ax
            break
    handles, labels = first_ax.get_legend_handles_labels()

    if legend_ncol is None:
        legend_ncol = min(max(2, len(labels)), 5)

    if legend_loc == "bottom":
        fig.legend(
            handles, labels,
            loc="center",
            bbox_to_anchor=(0.82, -0.065),  # Bottom outside
            ncol=legend_ncol,
            frameon=False,
            fontsize=14
        )
    elif legend_loc == "top":
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=legend_ncol,
            frameon=False,
        )
    else:  # right
        fig.legend(
            handles, labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=1,
            frameon=False,
        )

    # Global labels
    fig.supxlabel(x_label, y=-0.085)
    fig.supylabel(y_label, x=-0.01)

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, axes
