import numpy as np
import networkx as nx
from shapely.geometry import LineString
from networkx.algorithms.simple_paths import shortest_simple_paths

def monte_carlo_smoothed_path_selection(G, source, target, polygons_gdf, alpha, k=5):
    """
    Zoek k gridpaden van source naar target, smooth ze en kies het beste smoothed pad.
    """
    best_path_edges = None
    best_score = float("inf")

    apply_weighted_cost(G, alpha)

    try:
        paths = shortest_simple_paths(G, source, target, weight='cost')
        for i, path_nodes in enumerate(paths):
            if i >= k:
                break

            smoothed_edges = smooth_grid_path_within_polygon(G, path_nodes, polygons_gdf)
            total_cost = sum(e['length'] for e in smoothed_edges)  # of bijv. risk*length

            if total_cost < best_score:
                best_score = total_cost
                best_path_edges = smoothed_edges

    except nx.NetworkXNoPath:
        return None

    return best_path_edges

def monte_carlo_connect_all(G, polygons_gdf, alpha=0.5, k=5):
    distribution_nodes = [n for n, attr in G.nodes(data=True) if attr.get('ntype') == 'distribution']
    postnl_nodes = [n for n, attr in G.nodes(data=True) if attr.get('ntype') == 'postnl']

    results = []
    failures = []

    for postnl_node in postnl_nodes:
        best_path = None
        best_dist = float("inf")
        for dist_node in distribution_nodes:
            smoothed = monte_carlo_smoothed_path_selection(G, dist_node, postnl_node, polygons_gdf, alpha, k=k)
            if smoothed:
                total = sum(e['length'] for e in smoothed)
                if total < best_dist:
                    best_dist = total
                    best_path = (dist_node, postnl_node, total, smoothed)
        
        if best_path:
            results.append(best_path)
            print(f"Connected: {best_path[0]} → {postnl_node} | {best_dist:.1f} m")
        else:
            failures.append(postnl_node)
            print(f"No smoothed path: {postnl_node}")

    print(f"\nSummary:")
    print(f" - Successful: {len(results)}")
    print(f" - Failed:     {len(failures)}")

    return results, failures

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

def smooth_grid_path_within_polygon(G, path, polygons_gdf):
    smoothed_path = []
    i = 0

    while i < len(path) - 1:
        current = path[i]
        pt_i = G.nodes[current]['geometry']
        area_type = G.edges[current, path[i+1]]['etype']
        polygon = polygons_gdf[polygons_gdf['area_type'] == area_type].geometry.unary_union

        max_j = i + 1
        for j in range(i + 2, len(path)):
            line = LineString([G.nodes[current]['geometry'], G.nodes[path[j]]['geometry']])
            if polygon.contains(line):
                max_j = j
            else:
                break

        smoothed_path.append((current, path[max_j]))  # voeg superedge toe
        i = max_j

    smoothed_path.append((path[-2], path[-1]))  # voeg laatste stap toe

    # Optioneel: voeg edge attributen toe
    new_edges = []
    for u, v in smoothed_path:
        line = LineString([G.nodes[u]['geometry'], G.nodes[v]['geometry']])
        dist = line.length
        # Gemiddelde attributen ophalen van oorspronkelijke route (bijv. risico)
        new_edges.append({
            'u': u,
            'v': v,
            'geometry': line,
            'length': dist,
            'risk': 0.5,  # of gemiddelde van oude
            'etype': 'smoothed',
            'height': 30
        })

    return new_edges
