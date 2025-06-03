import networkx as nx
import numpy as np
from shapely.geometry import Point, LineString
from shapely.wkt import loads as wkt_loads

def connect_to_nearest_edge(G, max_connection_distance=10):
    """
    Connects 'postnl' and 'distribution' nodes to their nearest edge within a specified distance.
    A new waypoint node is inserted on the closest edge, which is split into two, and the original node
    is connected to this new waypoint with a connector edge, height set to 0.
    """

    edge_lines = []
    edge_keys = []

    # Collect all edges and their geometries
    for u, v, data in G.edges(data=True):
        geom = data.get('geometry')
        if isinstance(geom, str):
            try: geom = wkt_loads(geom)
            except: continue
        if isinstance(geom, LineString):
            edge_lines.append(geom)
            edge_keys.append((u, v, data))

    print(f"Total edges indexed: {len(edge_lines)}")

    connected = 0
    skipped = 0

    # Iterate over all postnl and distribution nodes
    print("Start iterating over postnl and distribution nodes.")
    for node_id, node_data in list(G.nodes(data=True)):
        node_type = node_data.get("ntype")
        if node_type not in ("postnl", "distribution"):
            continue

        geom = node_data.get("geometry")
        if isinstance(geom, str):
            try: geom = wkt_loads(geom)
            except: continue
        if not isinstance(geom, Point):
            continue

        # Find the closest edge
        min_dist = float("inf")
        best_projection = None
        best_edge = None

        for line, (u, v, data) in zip(edge_lines, edge_keys):
            if not line.is_valid or line.is_empty or not geom.is_valid or geom.is_empty:
                continue

            proj_dist = line.project(geom)
            if np.isnan(proj_dist):
                continue

            proj = line.interpolate(proj_dist)

            dist = geom.distance(proj)
            if dist < min_dist:
                min_dist = dist
                best_projection = proj
                best_edge = (u, v, data, line)

        if min_dist > max_connection_distance:
            print(f"Node {node_id} too far from any edge: {min_dist:.2f} m")
            skipped += 1
            continue

        u, v, edge_data, edge_geom = best_edge

        # Insert new node on the edge
        if G.has_edge(u, v):
            G.remove_edge(u, v)
        elif G.has_edge(v, u):
            G.remove_edge(v, u)
        else:
            print(f"⚠️ Edge ({u}, {v}) not found in graph. Skipping.")
            skipped += 1
            continue

        length_total = edge_geom.length
        length_u = Point(G.nodes[u]['geometry']).distance(best_projection)
        length_v = Point(G.nodes[v]['geometry']).distance(best_projection)

        G.add_node(best_projection, geometry=best_projection, ntype='waypoint', inserted=True)

        G.add_edge(u, best_projection, geometry=LineString([G.nodes[u]['geometry'], best_projection]),
                   length=length_u, risk=edge_data.get('risk', 0), etype=edge_data.get('etype', 'split'))
        G.add_edge(best_projection, v, geometry=LineString([best_projection, G.nodes[v]['geometry']]),
                   length=length_v, risk=edge_data.get('risk', 0), etype=edge_data.get('etype', 'split'))

        # Connect the postnl/distribution node to the new edge node
        dist_to_projection = geom.distance(best_projection)
        G.add_edge(node_id, best_projection, geometry=LineString([geom, best_projection]),
                   length=dist_to_projection, risk=0, etype='postnl_connector', height=0)

        print(f"Node {node_id} connected to edge ({u}-{v}) via new node {best_projection} at d = {min_dist:.2f} m")
        connected += 1

    print(f"\n Summary of connections:")
    print(f" - Connected: {connected}")
    print(f" - Skipped: {skipped}")

    return G
