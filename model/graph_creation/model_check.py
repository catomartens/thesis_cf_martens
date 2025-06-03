import networkx as nx
from collections import Counter
import geopandas as gpd
import pandas as pd

def diagnose_graph(G):
    print("GRAPH DIAGNOSTICS\n")

    # 1. Connectivity
    if nx.is_connected(G.to_undirected()):
        print("Graph is fully connected.")
    else:
        components = list(nx.connected_components(G.to_undirected()))
        component_sizes = [len(c) for c in components]
        print(f"Graph is NOT fully connected. Components: {len(components)}")
        print(f"Component size distribution: {Counter(component_sizes)}")
    # 2. Node ntypes
    node_ntypes = Counter(nx.get_node_attributes(G, "ntype").values())
    print("\nNode types (ntype):")
    for ntype, count in node_ntypes.items():
        print(f" - {ntype}: {count}")

    # 3. Edge etypes
    edge_etypes = Counter(nx.get_edge_attributes(G, "etype").values())
    print("\nEdge types (etype):")
    for etype, count in edge_etypes.items():
        print(f" - {etype}: {count}")

    # 4. Missing node attributes
    node_missing = {
        "geometry": 0,
        "ntype": 0,
    }
    for n, data in G.nodes(data=True):
        for key in node_missing:
            if key not in data or data[key] is None:
                node_missing[key] += 1
    print("\n Missing node attributes:")
    for key, count in node_missing.items():
        print(f" - {key}: {count} missing")

    # 5. Missing edge attributes
    edge_missing = {
        "geometry": 0,
        "length": 0,
        "risk": 0,
        "height": 0,
        "etype": 0,
    }
    for u, v, data in G.edges(data=True):
        for key in edge_missing:
            if key not in data or data[key] is None:
                edge_missing[key] += 1
    print("\nMissing edge attributes:")
    for key, count in edge_missing.items():
        print(f" - {key}: {count} missing")

    print("\nDiagnostics complete.")

def fill_missing_edge_heights_from_csv(G):
    """
    Fills missing edge 'height' values in the graph using a reference mapping from a CSV file.

    Args:
        G (networkx.Graph): The input graph with edges that may have missing 'height' attributes.

    Returns:
        networkx.Graph: The updated graph with missing edge heights filled in.
    """
    etype_col = "area_type"
    height_col = "Height"

    # Step 1: Load height map from CSV
    df = pd.read_csv('/Users/cmartens/Documents/thesis_cf_martens/model/graph_creation/input/all_edge_types_with_properties.csv')
    height_map = (
        df[[etype_col, height_col]]
        .dropna()
        .drop_duplicates()
        .set_index(etype_col)[height_col]
        .to_dict()
    )

    print(f"Height map loaded from CSV: {height_map}")

    # Step 2: Fill missing heights
    patched = 0
    for u, v, data in G.edges(data=True):
        if "height" not in data or data["height"] is None:
            etype = data.get("etype")
            if etype in height_map:
                data["height"] = height_map[etype]
                patched += 1
                print(f"Patched edge ({u} → {v}) with height={height_map[etype]} from etype='{etype}'")
            else:
                print(f"Cannot patch edge ({u} → {v}): etype='{etype}' not found in CSV")

    print(f"\nTotal edges patched: {patched}")

    print(df['Height'])
    return G



