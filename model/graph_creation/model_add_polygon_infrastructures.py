import geopandas as gpd
import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString, box
from shapely.ops import split
from shapely.strtree import STRtree
from shapely.wkt import loads as wkt_loads
from scipy.spatial import cKDTree
from collections import defaultdict

def add_grid_structure_to_polygons(G, gdf_polygons, grid_size=50, max_cells=10000):
    """
    Add a grid structure to polygons in the GeoDataFrame and connect them to the graph.

    Args:
        G (networkx.Graph): The graph to which the grid structure will be added.
        gdf_polygons (geopandas.GeoDataFrame): The GeoDataFrame containing the polygons.
        grid_size (int): The size of the grid cells.
        max_cells (int): Maximum number of cells to add.

    Returns:
        networkx.Graph: The updated graph with the grid structure added.
    """

    if gdf_polygons.empty:
        print("No polygons found in the GeoDataFrame.")
        return G
    
    print(f"Adding grid structure to {len(gdf_polygons)} polygons, with grid size {grid_size} and max cells {max_cells}.")
    for idx, row in gdf_polygons.iterrows():
        polygon = row.geometry
        risk = row.get("risk")
        area_type = row.get("area_type")
        height = row.get("Height")

        if area_type == "No-fly zone": # skip no-fly zones because they dont need a grid
            continue

        minx, miny, maxx, maxy = polygon.bounds

        x_coords = np.arange(minx, maxx, grid_size)
        y_coords = np.arange(miny, maxy, grid_size)

        nodes = {}
        cells_added = 0

        for x in x_coords:
            for y in y_coords:
                cell = box(x, y, x + grid_size, y + grid_size)
                if not cell.intersects(polygon):
                    continue

                inter = cell.intersection(polygon)
                if inter.is_empty or not inter.is_valid or inter.area < 1:
                    continue

                cx, cy = inter.centroid.x, inter.centroid.y
                node_key = (round(cx, 3), round(cy, 3))
                if node_key not in G:
                    G.add_node(node_key, geometry=Point(cx, cy), ntype='grid_point')
                    nodes[(x, y)] = node_key

                cells_added += 1
                if cells_added > max_cells:
                    print(f"Polygon {idx} exceeded max cells ({max_cells}).")
                    break
            if cells_added > max_cells:
                break

            # Connect neighbors nodes 
            for (x, y), node in nodes.items():
                for dx, dy in [(grid_size, 0), (0, grid_size)]:
                    neighbor_coord = (x + dx, y + dy)
                    neighbor = nodes.get(neighbor_coord)

                    if neighbor:
                        pt1 = G.nodes[node]['geometry']
                        pt2 = G.nodes[neighbor]['geometry']
                        line = LineString([pt1, pt2])
                        dist = pt1.distance(pt2)

                        if line.crosses(polygon.boundary):
                            continue  

                        G.add_edge(node, neighbor, geometry=line, risk=risk, length=dist, etype=area_type, height=height)

    return G

def add_local_polygon_boundaries_to_graph(G, gdf_polygons, snap_tolerance=35, boundary_sample_distance=50):
    """
    Adds local polygon boundaries to the graph. Connects grid points to the nearest boundary point.

    Args:
        G (networkx.Graph): The graph to which the polygon boundaries will be added.
        gdf_polygons (geopandas.GeoDataFrame): The GeoDataFrame containing the polygons.
        snap_tolerance (float): Tolerance for snapping points to the nearest boundary point.
        boundary_sample_distance (float): Distance between sampled points on the polygon boundary.

    Returns:
        networkx.Graph: The updated graph with the connected polygon boundaries added.
    """

    print(f"Adding local polygon boundaries to the graph with snap tolerance {snap_tolerance} and boundary sample distance {boundary_sample_distance}.")
    # iterate over polygons and add them to the graph
    for idx, row in gdf_polygons.iterrows():
        polygon = row.geometry
        risk = row.get("risk", 0)
        area_type = row.get("area_type")

        if area_type == "No-fly zone":
            continue

        boundary = polygon.exterior
        length = boundary.length
        num_samples = max(int(length // boundary_sample_distance), 3)
        sampled_points = [boundary.interpolate(i / num_samples, normalized=True) for i in range(num_samples + 1)]

        # Add boundary points to the graph
        boundary_node_keys = []
        for pt in sampled_points:
            key = (round(pt.x, 3), round(pt.y, 3))
            if key not in G:
                G.add_node(key, geometry=pt, ntype='polygon_boundary')
            boundary_node_keys.append(key)

        # Add polygon boundary edges to the graph
        for i in range(len(boundary_node_keys) - 1):
            k1, k2 = boundary_node_keys[i], boundary_node_keys[i + 1]
            dist = G.nodes[k1]['geometry'].distance(G.nodes[k2]['geometry'])
            line = LineString([G.nodes[k1]['geometry'], G.nodes[k2]['geometry']])
            G.add_edge(k1, k2, geometry=line, risk=risk, etype=area_type, length=dist, height=row.get("Height"))
        if len(boundary_node_keys) > 2:
            k1, k2 = boundary_node_keys[-1], boundary_node_keys[0]
            dist = G.nodes[k1]['geometry'].distance(G.nodes[k2]['geometry'])
            line = LineString([G.nodes[k1]['geometry'], G.nodes[k2]['geometry']])
            G.add_edge(k1, k2, geometry=line, risk=risk, etype=area_type, length=dist, height=row.get("Height"))

        # Find only grid nodes within the polygon
        grid_nodes = [(k, data['geometry']) for k, data in G.nodes(data=True)
                      if data.get("ntype") == "grid_point" and polygon.contains(data["geometry"])]
        if not grid_nodes:
            continue

        # Spatially index the boundary points
        boundary_coords = np.array([[pt.x, pt.y] for pt in sampled_points])
        boundary_keys = boundary_node_keys
        tree = cKDTree(boundary_coords)

        # Connect grid points to the nearest boundary point
        for key, pt in grid_nodes:
            dist, idx = tree.query([pt.x, pt.y], distance_upper_bound=snap_tolerance)
            if idx < len(boundary_keys):
                boundary_key = boundary_keys[idx]
                line = LineString([pt, G.nodes[boundary_key]['geometry']])
                dist = pt.distance(G.nodes[boundary_key]['geometry'])
                G.add_edge(key, boundary_key, geometry=line, risk=risk, height=row.get("Height"), length=dist, etype=area_type)

    return G

def snap_intersections_with_edges(intersections, tolerance=3):
    """ Clusters nearby intersection points within a given tolerance and returns representative points with the set of edges they intersect, for later splitting.

    Args:
        intersections (list): A list of tuples containing intersection points and the corresponding edge pairs.
        tolerance (float): The snapping tolerance.
    Returns:
        list: A list of tuples containing snapped intersection points and the corresponding edge pairs.
    """

    coords = np.array([[pt.x, pt.y] for pt, _, _ in intersections])
    tree = cKDTree(coords)
    groups = tree.query_ball_tree(tree, r=tolerance)

    edge_map = {}
    for group in groups:
        rep_coord = tuple(np.round(coords[group[0]], 3))
        if rep_coord not in edge_map:
            edge_map[rep_coord] = set()
        for idx in group:
            _, edge1, edge2 = intersections[idx]
            edge_map[rep_coord].add(frozenset(edge1))
            edge_map[rep_coord].add(frozenset(edge2))

    result = [(Point(xy), list(edge_map[xy])) for xy in edge_map]
    return result

def find_intersections_with_edges(G):
    """
    Finds all intersection points between edges in the graph and returns them along with the corresponding edge pairs for each intersection.

    Args:
        G (networkx.Graph): The graph to analyze.

    Returns:
        list: A list of tuples containing intersection points and the corresponding edge pairs.
    """
    edge_geoms = []
    edge_keys = []

    print(f"Finding intersections for {G.number_of_edges()} edges.")

    for u, v, data in G.edges(data=True):
        geom = data.get("geometry")
        if isinstance(geom, LineString):
            edge_geoms.append(geom)
            edge_keys.append((u, v))

    # Spatial index for edges
    tree = STRtree(edge_geoms)
    intersections = []

    for i, geom_i in enumerate(edge_geoms):
        for j in tree.query(geom_i):
            if i >= j:
                continue

            geom_j = edge_geoms[j]
            try:
                inter = geom_i.intersection(geom_j)
            except Exception:
                continue

            if inter.is_empty:
                continue

            pair1 = edge_keys[i]
            pair2 = edge_keys[j]

            if inter.geom_type == 'Point':
                intersections.append((inter, pair1, pair2))
            elif inter.geom_type == 'MultiPoint':
                for pt in inter.geoms:
                    intersections.append((pt, pair1, pair2))

    return intersections

def connect_intersections_to_polygon_boundaries(G, max_connection_distance=15):
    """
    Connects line_intersection and waypoint nodes to the nearest polygon_boundary node
    within a given distance, creating connector edges where appropriate.
    Important to conncect the linestring builded graph to the polygon builded graph.

    Args:
        G (networkx.Graph): The graph to which the connections will be added.
        max_connection_distance (float): The maximum distance for connecting nodes.

    Returns:
        networkx.Graph: The updated graph with the connections added.
    """

    # Collect all polygon boundary nodes and their geometries
    boundary_nodes = []
    boundary_geoms = []

    for node_id, node_data in G.nodes(data=True):
        if node_data.get("ntype") == "polygon_boundary":
            geom = node_data.get("geometry")
            if isinstance(geom, Point):
                boundary_nodes.append(node_id)
                boundary_geoms.append(geom)

    if not boundary_geoms:
        print("No polygon boundary points found.")
        return G

    # Build spatial index for fast proximity queries
    print(f"STRtree indexing over {len(boundary_geoms)} polygon_boundary points.")
    tree = STRtree(boundary_geoms)
    geom_to_node = {id(geom): node_id for geom, node_id in zip(boundary_geoms, boundary_nodes)}

    connected = 0
    skipped = 0

    # Iterate over all intersection and waypoint nodes
    for node_id, node_data in list(G.nodes(data=True)):
        node_type = node_data.get("ntype")
        if node_type not in ("line_intersection", "waypoint"):
            continue

        geom = node_data.get("geometry")
        if isinstance(geom, str):
            try:
                geom = wkt_loads(geom)
            except:
                continue
        if not isinstance(geom, Point):
            continue

        # Find boundary nodes within connection distance
        candidate_idxs = tree.query(geom.buffer(max_connection_distance))
        candidates = [boundary_geoms[i] for i in candidate_idxs]
        nearby_geoms = [g for g in candidates if geom.distance(g) <= max_connection_distance]

        # Find the closest valid boundary point
        min_dist = float("inf")
        best_target_id = None
        best_point = None

        for b_geom in nearby_geoms:
            dist = geom.distance(b_geom)
            if dist < min_dist:
                min_dist = dist
                best_point = b_geom
                best_target_id = geom_to_node.get(id(b_geom))

        if best_target_id is None:
            skipped += 1
            continue

        # Ensure boundary node exists in the graph
        key = (round(best_point.x, 3), round(best_point.y, 3))
        if key not in G:
            G.add_node(key, geometry=best_point, ntype='waypoint', inserted=True)

        # Add connector edge if not already present
        if not G.has_edge(node_id, key):
            G.add_edge(
                node_id, key,
                geometry=LineString([geom, best_point]),
                length=min_dist,
                risk=0,
                etype='connector',
                height=30
            )
            connected += 1

    print(f"Connected nodes: {connected}")

    if skipped > 0:
        print(f"Skipped {skipped} line intersections/waypoints that could not be connected to polygon boundaries. Consider increasing the max_connection_distance.")

    return G

def add_grid_polygons_to_graph(G, polygons_gdf, lines_gdf):
    print("Excecuting add_grid_polygons_to_graph function")
    print(f"G before polygons: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # Add grid structure to polygons
    print("\n Step 1")
    G = add_grid_structure_to_polygons(G, polygons_gdf, grid_size=50)
    print(f"G after grid: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # Add boundaries to polygons
    print("\n Step 2")
    G = add_local_polygon_boundaries_to_graph(G, polygons_gdf)

    # Find intersections between lines
    print("\n Step 3")
    intersections_raw = find_intersections_with_edges(G)
    print(f"Found {len(intersections_raw)} intersections between polygon-polygon and polygon-line.")

    print("\n Step 4")
    # Snap intersections to the nearest point within the tolerance
    intersections = snap_intersections_with_edges(intersections_raw, tolerance=2)
    print(f"Number of unique snapt points: {len(intersections)}")
    
    # Add lines to the graph from intersections
    print("\n Step 5")
    print("Mapping geometries to edges.")
    geom_to_uv = {}
    for u, v, data in G.edges(data=True):
        geom = data.get('geometry')
        if isinstance(geom, LineString):
            key = frozenset([u, v])
            geom_to_uv[key] = (u, v, data)

    # Adding intersection points to the graph
    print('\n Step 6')
    print("Adding intersection nodes.")
    added_nodes = 0
    for pt, _ in intersections:
        key = (round(pt.x, 3), round(pt.y, 3))
        if key not in G:
            G.add_node(key, geometry=pt, ntype='poly_intersection', risk=0)
            added_nodes += 1

    print(f"Intersection nodes added: {added_nodes}")

    # Split edges at intersections
    print("\n Step 7")
    print(" Splitting edges at intersections.")
    added_edges = 0

    for pt, edge_keys in intersections:
        split_node = (round(pt.x, 3), round(pt.y, 3))
        if split_node not in G:
            G.add_node(split_node, geometry=pt, ntype='poly_intersection', risk=0)

        for edge_key in edge_keys:
            u, v = tuple(edge_key)
            if not G.has_edge(u, v) and not G.has_edge(v, u):
                continue

            # Make consistent ordering of u and v
            if not G.has_edge(u, v):
                u, v = v, u

            data = G.get_edge_data(u, v)
            if data is None:
                continue

            G.remove_edge(u, v) # remove the original edge

            pu = G.nodes[u]['geometry']
            pv = G.nodes[v]['geometry']
            p_split = G.nodes[split_node]['geometry']

            if pu.distance(p_split) > 0:
                G.add_edge(u, split_node,
                        geometry=LineString([pu, p_split]),
                        length=pu.distance(p_split),
                        risk=data.get('risk'),
                        etype=data.get('etype', 'split'),
                        height=data.get('height'))

            if p_split.distance(pv) > 0:
                G.add_edge(split_node, v,
                        geometry=LineString([p_split, pv]),
                        length=p_split.distance(pv),
                        risk=data.get('risk'),
                        etype=data.get('etype', 'split'),
                        height=data.get('height'))


    print(f" Edges split and added: {added_edges}")

    print("\n Step 8")
    G = connect_intersections_to_polygon_boundaries(G, max_connection_distance=15)

    print(f"G after connecting: {len(G.nodes)} nodes, {len(G.edges)} edges")

    return G
