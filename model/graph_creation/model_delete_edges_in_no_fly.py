import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString
from shapely.strtree import STRtree

def extract_edges_to_gdf(G):
    """
    Extracts all valid LineString edges from a graph and returns them as a GeoDataFrame,
    including geometry, length, and risk attributes. Only edges with LineString geometries are included.
    """

    edges = []

    # Iterate over all edges in the graph
    for u, v, data in G.edges(data=True):
        geom = data.get('geometry')

        # Skip edges without a valid LineString geometry
        if not isinstance(geom, LineString):
            continue

        # Store edge attributes in a dictionary
        edges.append({
            'geometry': geom,
            'length': data.get('length', geom.length),  # fallback to geometry length if missing
            'risk': data.get('risk', None)
        })

    # Return as GeoDataFrame with EPSG:28992 (Dutch RD New) coordinate system
    return gpd.GeoDataFrame(edges, geometry='geometry', crs="EPSG:28992")

def get_lines_inside_no_fly_zones(edges_gdf, no_fly_zones):
    """
    Identifies and returns all edges from the input GeoDataFrame that are located inside,
    cross, touch, or intersect with any no-fly zone polygons.

    Args:
        edges_gdf (GeoDataFrame): GeoDataFrame containing LineString edges.
        polygons_gdf (GeoDataFrame): GeoDataFrame containing polygon features, including no-fly zones.

    Returns:
        GeoDataFrame: Subset of edges_gdf that intersect or fall within no-fly zones.
    """

    if no_fly_zones.empty:
        print("No no-fly zones found.")
        return gpd.GeoDataFrame(columns=edges_gdf.columns, crs=edges_gdf.crs)

    print(f"Checking against {len(no_fly_zones)} no-fly zones.")

    # Build spatial index for fast intersection queries
    zones = list(no_fly_zones.geometry)
    tree = STRtree(zones)

    to_remove = []

    # Check each edge for intersection with any no-fly zone
    for idx, row in edges_gdf.iterrows():
        geom = row["geometry"]
        if not isinstance(geom, LineString):
            continue

        candidate_idxs = tree.query(geom)

        for i in candidate_idxs:
            zone_geom = zones[i]
            if (geom.within(zone_geom) or
                geom.crosses(zone_geom) or
                geom.touches(zone_geom) or
                geom.intersects(zone_geom)):
                to_remove.append(idx)
                break

    result = edges_gdf.loc[to_remove].copy()
    print(f"{len(result)} edges marked for removal (within no-fly zones).")

    return result

def remove_filtered_edges_from_graph(G, filtered_edges_gdf):
    """
    Removes all edges from the graph G whose geometry exactly matches
    any geometry in the filtered_edges_gdf.

    Parameters:
        G (networkx.Graph): The input graph.
        filtered_edges_gdf (GeoDataFrame): A GeoDataFrame containing edges to remove (based on geometry match).

    Returns:
        networkx.Graph: The updated graph with matching edges removed.
    """

    # Convert geometries to WKB for fast and precise comparison
    filtered_wkbs = set(filtered_edges_gdf["geometry"].apply(lambda g: g.wkb))

    to_remove = []

    # Identify matching edges
    for u, v, data in G.edges(data=True):
        geom = data.get("geometry")
        if isinstance(geom, LineString) and geom.wkb in filtered_wkbs:
            to_remove.append((u, v))

    # Remove matched edges from the graph
    G.remove_edges_from(to_remove)
    print(f"{len(to_remove)} edges removed from the graph.")

    return G

def remove_no_fly_zones_from_graph(G, no_fly_zones):
    """
    Removes all edges from G that fall within or intersect with no-fly zones.

    Parameters:
        G (networkx.Graph): The input graph.
        no_fly_zones_gdf (GeoDataFrame): Only the no-fly zone polygons (already filtered).

    Returns:
        networkx.Graph: The updated graph with restricted edges removed.
    """

    print("Get all lines from the graph.")
    all_lines = extract_edges_to_gdf(G)

    if all_lines.empty:
        print("No lines found in the graph.")
        return G

    print("Get all lines that are inside no-fly zones.") 
    filtered_edges = get_lines_inside_no_fly_zones(all_lines, no_fly_zones)

    print(f"Number of edges inside no-fly zones: {len(filtered_edges)}")

    print("Removing filtered edges from the graph.")
    G_filtered = remove_filtered_edges_from_graph(G, filtered_edges)

    return G_filtered
