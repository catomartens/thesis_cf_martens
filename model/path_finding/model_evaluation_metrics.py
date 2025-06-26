import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx

from model_find_paths_metrics import connect_distribution_to_postnl

def evaluate_alpha_tradeoff(G, alpha_list, method):
    """
    Evaluate routing trade-offs for different alpha values by computing summary metrics
    across all distribution → PostNL paths using the metrics_df from connect_distribution_to_postnl.

    Returns:
        tuple:
            - pd.DataFrame: Summary per alpha with aggregated metrics
            - list of str: All etypes observed across all alphas (flat list)
            - dict: Mapping from alpha to the list of etypes for that alpha
    """
    results = []
    all_etypes = []
    etypes_per_alpha = {}

    for alpha in alpha_list:
        # Get metrics_df directly from connect_distribution_to_postnl
        connected, not_connected, metrics_df = connect_distribution_to_postnl(G, alpha, method=method)
        
        # Collect etypes for this alpha
        etypes_this_alpha = []
        for _, _, _, path_geometries, etype_array in connected:
            etypes_this_alpha.extend(etype_array)
        
        all_etypes.extend(etypes_this_alpha)
        etypes_per_alpha[alpha] = etypes_this_alpha
        
        # Calculate summary statistics from metrics_df
        if not metrics_df.empty:
            result = {
                "alpha": alpha,
                "n_paths": len(metrics_df),
                "mean_length": metrics_df['length'].mean(),
                "min_length": metrics_df['length'].min(),
                "max_length": metrics_df['length'].max(),
                "std_length": metrics_df['length'].std(),
                "mean_risk": metrics_df['risk'].mean(),
                "min_risk": metrics_df['risk'].min(),
                "max_risk": metrics_df['risk'].max(),
                "std_risk": metrics_df['risk'].std(),
                "mean_energy": metrics_df['energy'].mean(),
                "min_energy": metrics_df['energy'].min(),
                "max_energy": metrics_df['energy'].max(),
                "std_energy": metrics_df['energy'].std(),
                "mean_turns": metrics_df['turns'].mean(),
                "min_turns": metrics_df['turns'].min(),
                "max_turns": metrics_df['turns'].max(),
                "mean_height_changes": metrics_df['height_changes'].mean(),
                "min_height_changes": metrics_df['height_changes'].min(),
                "max_height_changes": metrics_df['height_changes'].max(),
                "n_unique_etypes": len(set(etypes_this_alpha)),
                "n_failed_connections": len(not_connected)
            }
        else:
            # No successful connections
            result = {
                "alpha": alpha,
                "n_paths": 0,
                "mean_length": np.nan,
                "min_length": np.nan,
                "max_length": np.nan,
                "std_length": np.nan,
                "mean_risk": np.nan,
                "min_risk": np.nan,
                "max_risk": np.nan,
                "std_risk": np.nan,
                "mean_energy": np.nan,
                "min_energy": np.nan,
                "max_energy": np.nan,
                "std_energy": np.nan,
                "mean_turns": np.nan,
                "min_turns": np.nan,
                "max_turns": np.nan,
                "mean_height_changes": np.nan,
                "min_height_changes": np.nan,
                "max_height_changes": np.nan,
                "n_unique_etypes": 0,
                "n_failed_connections": len(not_connected)
            }
        
        results.append(result)
        
        # Print summary for this alpha
        print(f"\nAlpha {alpha:.2f} summary:")
        print(f"  Successful paths: {result['n_paths']}")
        print(f"  Failed connections: {result['n_failed_connections']}")
        if result['n_paths'] > 0:
            print(f"  Avg length: {result['mean_length']:.1f} m")
            print(f"  Avg risk: {result['mean_risk']:.1f}")
            print(f"  Avg energy: {result['mean_energy']:.1f} Wh")
            print(f"  Avg turns: {result['mean_turns']:.1f}")
            print(f"  Avg height changes: {result['mean_height_changes']:.1f}")
    
    return pd.DataFrame(results), all_etypes, etypes_per_alpha

def plot_common_etypes_per_alpha(etype_dict_per_alpha, city="Unknown", title_prefix="Most Common Etypes in"):
    """
    Creates one bar plot per alpha, showing the frequency of etypes.
    Ensures the same y-axis scale across all plots for comparison.

    Args:
        etype_dict_per_alpha (dict): Dictionary where keys are alpha values and values are lists of etypes.
        title_prefix (str): Title prefix to display before the alpha value.
    """

    area_color_map = {
        # Infrastructuur
        'Motorways and major roads': '#E31A1C',       # felrood
        'Regional roads': '#F46D43',                  # vurig oranje
        'Tracks and rural access roads': '#D95F02',   # donkeroranje
        'Living and residential streets': '#FFCB05',  # felgeel
        'Pedestrian and cycling paths': '#66C2A5',    # turquoisegroen
        'Railways': '#4B4B4B',                        # donkergrijs
        'Helipads': '#F0027F',
        'Airports and airfields': '#A6CEE3',

        # Energie & communicatie
        'Power lines': '#999999',
        'Power plants': '#FF8C00',
        'Communication towers': '#FB9A99',
        'High infrastructures': '#888888',

        # Gebouwd gebied
        'Industrial zones': '#377EB8',               # helderblauw
        'Commercial zones': '#FFC300',               # goudgeel
        'Retail zones': '#FF8C00',                   # warm oranje
        'Residential areas': '#FFA07A',              # warm zalm/oranjerood – beter onderscheid met 'Retail zones'
        'Schools and universities': '#DAA520',
        'Hospitals': '#DC143C',
        'Care homes': '#FF9999',
        'Prisons': '#696969',
        'Religious sites': '#C71585',
        'Cemeteries': '#556B2F',
        'Cultural sites': '#8A2BE2',

        # Natuur
        'Parks': '#7CFC00',                          # limegroen
        'Recreational zones': '#20B2AA',
        'Meadows and open grass': '#9ACD32',         # geelgroen
        'Forests and woodlands': '#228B22',          # bosgroen
        'Agricultural lands': '#BFD92E',             # helderder dan '#ADDE87'
        'Wetlands': '#66B2FF',                       # licht, helder blauw
        'Lakes and ponds': '#1E90FF',                # diepblauw (meer contrast met 'Wetlands')
        'Water reservoirs': '#4682B4',
        'Rivers, canals and streams': '#5DADE2',
        'Natura2000 areas': '#005824',

        # Overig
        'connector': '#D3D3D3',
        'No-fly zone': '#B22222'
    }
    all_etypes = sorted(set(et for etypes in etype_dict_per_alpha.values() for et in etypes))
    global_max = max((Counter(ets).most_common(1)[0][1] if ets else 0) for ets in etype_dict_per_alpha.values())

    for alpha, etypes in etype_dict_per_alpha.items():
        counts = Counter(etypes)
        etype_df = pd.DataFrame({
            "etype": all_etypes,
            "count": [counts.get(e, 0) for e in all_etypes]
        })

        plt.figure(figsize=(12, 5))
        colors = [area_color_map.get(e, "#cccccc") for e in etype_df["etype"]]
        sns.barplot(data=etype_df, x="etype", y="count", palette=colors)
        plt.ylim(0, global_max * 1.1)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Edge type (etype)")
        plt.ylabel("Total count")
        plt.title(f"{title_prefix} {city} (α = {alpha:.2f})")
        plt.tight_layout()
        plt.show()

def plot_common_etypes_grouped(etype_dict_per_alpha, city="Unknown", title_prefix="Etype Comparison"):
    """
    Creates a grouped bar plot comparing etype counts between alpha = 0 and alpha = 1.

    Args:
        etype_dict_per_alpha (dict): Dictionary where keys are alpha values (e.g., 0.0, 1.0) and values are lists of etypes.
        city (str): Name of the city for plot title.
        title_prefix (str): Custom title prefix.
    """

    # Only proceed if we have both alpha = 0 and alpha = 1
    if 0.0 not in etype_dict_per_alpha or 1.0 not in etype_dict_per_alpha:
        raise ValueError("etype_dict_per_alpha must contain both alpha=0.0 and alpha=1.0")

    # Get all etypes present
    all_etypes = sorted(set(etype_dict_per_alpha[0.0] + etype_dict_per_alpha[1.0]))

    # Build a dataframe
    data = []
    for alpha in [0.0, 1.0]:
        counts = Counter(etype_dict_per_alpha[alpha])
        for etype in all_etypes:
            data.append({
                "etype": etype,
                "count": counts.get(etype, 0),
                "alpha": f"α = {alpha:.0f}"
            })

    df = pd.DataFrame(data)

    # Plot grouped bar chart
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df, x="etype", y="count", hue="alpha")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Edge type (etype)")
    plt.ylabel("Total count")
    plt.title(f"{title_prefix} for {city}")
    plt.legend(title="Alpha")
    plt.tight_layout()
    plt.show()

def plot_pareto_tradeoff(df_summary, x_metric="mean_risk", y_metric="mean_energy"):
    """
    Plot Pareto frontier for trade-off between risk and energy with per-alpha color-coded legend.

    Args:
        df_summary (pd.DataFrame): DataFrame from evaluate_alpha_tradeoff.
        x_metric (str): Column name for x-axis (e.g., "mean_risk").
        y_metric (str): Column name for y-axis (e.g., "mean_energy").
    """
    plt.figure(figsize=(8, 6))

    # Create color mapping
    cmap = cm.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=min(df_summary["alpha"]), vmax=max(df_summary["alpha"]))
    alpha_colors = {alpha: cmap(norm(alpha)) for alpha in df_summary["alpha"]}

    # Plot each point individually with its alpha color
    for _, row in df_summary.iterrows():
        plt.scatter(
            row[x_metric],
            row[y_metric],
            color=alpha_colors[row["alpha"]],
            s=100
        )
        plt.text(
            row[x_metric] + 0.01,
            row[y_metric],
            f"α = {row['alpha']:.2f}",
            fontsize=9,
            ha='left',
            va='top'
        )

    # Create legend handles
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f"α = {alpha:.2f}",
               markerfacecolor=color, markersize=10)
        for alpha, color in alpha_colors.items()
    ]

    plt.legend(handles=legend_elements, title="Alpha (α)", loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.xlabel("Average Total Risk per Path")
    plt.ylabel("Average Total Energy per Path")
    plt.title("Pareto Trade-off: Risk vs Energy (per α)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()