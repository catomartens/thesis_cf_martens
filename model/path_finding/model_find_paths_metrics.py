import numpy as np
import networkx as nx
import pandas as pd
from shapely.geometry import LineString

def apply_weighted_cost(
    G,
    alpha,
    default_energy_per_m=0.025,  # 25 Wh/km = 0.025 Wh/m
    energy_up_fixed=2.03,        # Fixed Wh for 30m ascent
    energy_down_fixed=0.93,      # Fixed Wh for 30m descent
    default_start_height=30
):
    """
    Applies a combined risk-energy cost to each edge in the graph.

    For vertical motion, a fixed energy penalty is added per 30m of altitude change.

    Args:
        G (networkx.Graph): The graph whose edges will be updated.
        alpha (float): Weighting factor between 0 (energy focus) and 1 (risk focus).
        default_energy_per_m (float): Energy consumption per meter for horizontal flight.
        energy_up_fixed (float): Fixed energy cost (Wh) for 30m ascent.
        energy_down_fixed (float): Fixed energy cost (Wh) for 30m descent.
        default_start_height (float): Default height when node height is missing.
    """

    for u, v, data in G.edges(data=True):
        risk = data.get('risk', 1)
        length = data.get('length', 1)
        area_type = data.get('area_type')

        if not isinstance(risk, (int, float)) or np.isnan(risk):
            print(f"Edge ({u}, {v}) has invalid 'risk' ({risk}) – defaulting to 1. Area type: {area_type}")
            risk = 1

        if not isinstance(length, (int, float)) or np.isnan(length):
            print(f"Edge ({u}, {v}) has invalid 'length' ({length}) – defaulting to 1. Area type: {area_type}")
            length = 1

        e_per_m = data.get('energy_per_m', default_energy_per_m)
        if not isinstance(e_per_m, (int, float)) or np.isnan(e_per_m):
            e_per_m = default_energy_per_m

        edge_height = data.get('height', default_start_height)
        if not isinstance(edge_height, (int, float)) or np.isnan(edge_height):
            print(f"Edge ({u}, {v}) has invalid 'height' ({edge_height}) – defaulting to {default_start_height}. Area type: {area_type}")
            edge_height = default_start_height

        start_height = G.nodes[u].get('height', default_start_height)

        horiz_energy = length * e_per_m
        delta_h = edge_height - start_height

        if delta_h > 0:
            vert_energy = energy_up_fixed
        elif delta_h < 0:
            vert_energy = energy_down_fixed
        else:
            vert_energy = 0

        total_energy = horiz_energy + vert_energy
        cost = alpha * (risk * length) + (1 - alpha) * total_energy

        data['cost'] = cost
        data['energy'] = total_energy

def euclidean_heuristic(u, v, G):
    """Heuristic function for A*: straight-line distance between node u and v."""
    p1 = G.nodes[u]['geometry']
    p2 = G.nodes[v]['geometry']
    return p1.distance(p2) 

def calculate_path_metrics(G, path_nodes, path_edges):
    """
    Calculate various metrics for a given path.
    
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    metrics = {
        'length': 0,
        'risk': 0,
        'energy': 0,
        'turns': 0,
        'height_changes': 0
    }
    
    # Previous direction for turn calculation
    prev_direction = None
    
    # Previous height for tracking changes
    prev_height = G.nodes[path_nodes[0]].get('height', 30)
    
    for i, (u, v) in enumerate(path_edges):
        edge_data = G.get_edge_data(u, v)
        
        # Length
        length = edge_data.get('length', 1)
        metrics['length'] += length
        
        # Risk (total risk = risk * length for the edge)
        risk = edge_data.get('risk', 1)
        metrics['risk'] += risk * length
        
        # Energy
        energy = edge_data.get('energy', 0)
        metrics['energy'] += energy
        
        # Height changes
        current_height = edge_data.get('height', 30)
        height_diff = current_height - prev_height
        
        # Count 30m changes (both up and down)
        if abs(height_diff) >= 30:
            metrics['height_changes'] += abs(height_diff) // 30
            
        prev_height = current_height
        
        # Direction changes (turns)
        if i > 0:
            # Get geometries to calculate direction
            geom = edge_data.get('geometry')
            if geom is None:
                p1 = G.nodes[u]['geometry']
                p2 = G.nodes[v]['geometry']
                geom = LineString([p1, p2])
            
            # Calculate direction vector
            coords = list(geom.coords)
            if len(coords) >= 2:
                dx = coords[-1][0] - coords[0][0]
                dy = coords[-1][1] - coords[0][1]
                current_direction = np.arctan2(dy, dx)
                
                if prev_direction is not None:
                    # Calculate angle difference
                    angle_diff = abs(current_direction - prev_direction)
                    # Normalize to [0, π]
                    if angle_diff > np.pi:
                        angle_diff = 2 * np.pi - angle_diff
                    
                    # Count as turn if angle > 30 degrees
                    if angle_diff > np.pi / 6:
                        metrics['turns'] += 1
                
                prev_direction = current_direction
    
    return metrics

def connect_distribution_to_postnl(G, alpha, method):
    """
    For each PostNL node, find the distribution node that yields the lowest total cost path.

    Args:
        G (networkx.Graph): Graph with nodes and weighted edges.
        alpha (float): Trade-off factor between risk and energy.
        method (str): 'astar' or 'dijkstra'

    Returns:
        connected (list of tuples): (distribution_node, postnl_node, path_nodes, path_edges, etype_array)
        not_connected (list): list of postnl_nodes that could not be connected.
        metrics_df (pd.DataFrame): DataFrame with metrics for each connection
    """
    if method not in ['astar', 'dijkstra']:
        raise ValueError("Method must be either 'astar' or 'dijkstra'")
        
    distribution_nodes = [n for n, attr in G.nodes(data=True) if attr.get('ntype') == 'distribution']
    postnl_nodes = [n for n, attr in G.nodes(data=True) if attr.get('ntype') == 'postnl']

    print(f"Distribution points: {len(distribution_nodes)}")
    print(f"PostNL points: {len(postnl_nodes)}")
    print(f"Alpha: {alpha}, Method: {method}")

    connected = []
    not_connected = []
    metrics_list = []

    apply_weighted_cost(G, alpha)

    for postnl_node in postnl_nodes:
        best_path = None
        best_total_cost = float('inf')
        best_metrics = None

        for dist_node in distribution_nodes:
            try:
                if method == 'astar':
                    path_nodes = nx.astar_path(G, dist_node, postnl_node,
                                               heuristic=lambda u, v=postnl_node: euclidean_heuristic(u, v, G),
                                               weight='cost')
                else:  # dijkstra
                    path_nodes = nx.shortest_path(G, dist_node, postnl_node, weight='cost')

                path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))

                # Calculate total cost for comparison
                total_cost = sum(G.get_edge_data(u, v).get('cost', 1) for u, v in path_edges)

                if total_cost < best_total_cost:
                    best_total_cost = total_cost
                    
                    # Get path geometries and edge types
                    path_geometries = []
                    etype_array = []
                    
                    for u, v in path_edges:
                        edge_data = G.get_edge_data(u, v)
                        geom = edge_data.get('geometry')
                        if geom is None:
                            p1 = G.nodes[u]['geometry']
                            p2 = G.nodes[v]['geometry']
                            geom = LineString([p1, p2])
                        
                        path_geometries.append((geom, u, v))
                        etype_array.append(edge_data.get('etype', 'unknown'))
                    
                    # Calculate metrics for this path
                    best_metrics = calculate_path_metrics(G, path_nodes, path_edges)
                    best_metrics['distribution_node'] = dist_node
                    best_metrics['postnl_node'] = postnl_node
                    
                    best_path = (dist_node, postnl_node, path_nodes, path_geometries, etype_array)

            except nx.NetworkXNoPath:
                continue

        if best_path:
            connected.append(best_path)
            metrics_list.append(best_metrics)
            print(f"Connected: {best_path[0]} → {postnl_node} | Length: {best_metrics['length']:.1f} m")
        else:
            not_connected.append((postnl_node,))
            print(f"No path: {postnl_node}")

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    
    # Calculate summary statistics
    print(f"\nConnection summary:")
    print(f" - Successful: {len(connected)}")
    print(f" - Failed:     {len(not_connected)}")
    
    if not metrics_df.empty:
        print(f"\nMetrics summary:")
        print(f" - Length:     min={metrics_df['length'].min():.1f}, max={metrics_df['length'].max():.1f}, avg={metrics_df['length'].mean():.1f} m")
        print(f" - Risk:       min={metrics_df['risk'].min():.1f}, max={metrics_df['risk'].max():.1f}, avg={metrics_df['risk'].mean():.1f}")
        print(f" - Energy:     min={metrics_df['energy'].min():.1f}, max={metrics_df['energy'].max():.1f}, avg={metrics_df['energy'].mean():.1f} Wh")
        print(f" - Turns:      min={metrics_df['turns'].min()}, max={metrics_df['turns'].max()}, avg={metrics_df['turns'].mean():.1f}")
        print(f" - Height Δ:   min={metrics_df['height_changes'].min()}, max={metrics_df['height_changes'].max()}, avg={metrics_df['height_changes'].mean():.1f}")

    return connected, not_connected, metrics_df