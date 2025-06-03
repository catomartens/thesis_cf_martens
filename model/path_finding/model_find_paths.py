import numpy as np
import networkx as nx
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

def connect_distribution_to_postnl(G, alpha, method):
    """
    Connect each PostNL node to its nearest distribution node via the shortest path.

    The cost is computed using 'cost' (combined risk and energy), weighted by alpha.

    Args:
        G (networkx.Graph): Graph with nodes and weighted edges.
        alpha (float): Trade-off factor between risk and energy.

    Returns:
        connected (list of tuples): (distribution_node, postnl_node, total_length, path_nodes, path_edges, etype_array)
        not_connected (list of tuples): (postnl_node,)
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

    apply_weighted_cost(G, alpha)

    for postnl_node in postnl_nodes:
        best_path = None
        best_cost = float('inf')
        best_data = None

        for dist_node in distribution_nodes:
            try:
                if method == 'astar':
                    path_nodes = nx.astar_path(G, dist_node, postnl_node, heuristic=lambda u, v=postnl_node: euclidean_heuristic(u, v, G), weight='cost')
                elif method == 'dijkstra':
                    path_nodes = nx.shortest_path(G, dist_node, postnl_node, weight='cost')
                
                path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))

                total_length = 1
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
                    total_length += edge_data.get('length', geom.length)

                if total_length < best_cost:
                    best_cost = total_length
                    best_path = (dist_node, postnl_node, total_length, path_nodes, path_geometries, etype_array)

            except nx.NetworkXNoPath:
                continue

        if best_path:
            connected.append(best_path)
            print(f"Connected: {best_path[0]} → {postnl_node} | {best_cost:.1f} m | {len(best_path[3])} nodes")
        else:
            not_connected.append((postnl_node,))
            print(f"No path: {postnl_node}")

    # Summary
    print(f"\nConnection summary:")
    print(f" - Successful: {len(connected)}")
    print(f" - Failed:     {len(not_connected)}")
    if connected:
        lengths = [c[2] for c in connected]
        print(f" - Average length: {np.mean(lengths):.1f} m")
        print(f" - Longest path:   {np.max(lengths):.1f} m")
        print(f" - Shortest path:  {np.min(lengths):.1f} m")

    return connected, not_connected

