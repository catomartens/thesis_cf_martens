import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import split
from shapely.strtree import STRtree
from collections import defaultdict
from scipy.spatial import cKDTree

def explode_multilines(gdf):
    """
    Explode MultiLineString geometries into separate LineString geometries.
    
    Args:
        gdf (geopandas.GeoDataFrame): The GeoDataFrame containing the geometries.
        """

    exploded = gpd.GeoDataFrame(
        gdf.explode(index_parts=False).reset_index(drop=True),
        crs=gdf.crs)
        
    return exploded

def find_intersections_per_line(lines_gdf):
    """
    Find intersections between lines in a GeoDataFrame using STR tree.

    Args:
        lines_gdf (geopandas.GeoDataFrame): The GeoDataFrame containing the line geometries.

    Returns:
        dict: A dictionary where keys are line indices and values are lists of intersection points.
    """
    if lines_gdf.empty:
        print("No lines found in the GeoDataFrame.")
        return {}

    # Create a list of lines and a spatial index
    lines = list(lines_gdf.geometry)
    tree = STRtree(lines)
    line_intersections = defaultdict(list)

    # Iterate through each line and find intersections with other lines
    for i, line in enumerate(lines):
        
        for j in tree.query(line):
            if i >= j:
                continue
            other = lines[j]
            try:
                inter = line.intersection(other)
            except Exception:
                continue
            if inter.is_empty:
                continue
            if inter.geom_type == 'Point':
                line_intersections[i].append(inter)
                line_intersections[j].append(inter)
            elif inter.geom_type == 'MultiPoint':
                for pt in inter.geoms:
                    line_intersections[i].append(pt)
                    line_intersections[j].append(pt)

    return line_intersections

def merge_close_points(points, tolerance=5):
    coords = np.array([[p[0], p[1]] for p in points])
    tree = cKDTree(coords)
    groups = tree.query_ball_tree(tree, r=tolerance)

    snapped = {}
    for group in groups:
        rep = tuple(coords[group[0]])
        for idx in group:
            snapped[tuple(coords[idx])] = rep

    return snapped


def add_linesstrings_to_graph(G, lines_gdf, snap_tolerance=5):
    
    # Get all all seperate exploded lines from the lines_gdf
    print("Exploding MultiLineString geometries into separate LineString geometries.")
    lines = explode_multilines(lines_gdf)

    if lines.empty:
        print("No lines found in the GeoDataFrame.")
        return G

    # Find all intersections per line
    print("Finding intersections between lines.")
    intersections_dict = find_intersections_per_line(lines)

    if not intersections_dict:
        print("No intersections found.")
        return G

    all_points = [pt for pts in intersections_dict.values() for pt in pts]
    print(f"Found {len(all_points)} intersection points.")

    # Snap intersection points to the nearest point within the tolerance
    snap_map = merge_close_points([(pt.x, pt.y) for pt in all_points], tolerance=snap_tolerance)
    snapped_points = [Point(snap_map.get((pt.x, pt.y), (pt.x, pt.y))) for pt in all_points]
    unique_snapped = set((round(p.x, 3), round(p.y, 3)) for p in snapped_points)

    # Add snapped points to the graph
    print("Adding snapped points to the graph.")
    added_nodes = 0

    for x, y in unique_snapped:
        if (x, y) not in G:
            G.add_node((x, y), geometry=Point(x, y), ntype='line_intersection', risk=0)
            added_nodes += 1

    added_edges = set()
    edge_count = 0
    new_waypoints = 0
    skipped_segments = 0
    short_segments = 0

    # Add intersection lines
    for i, row in lines.iterrows():
        if i not in intersections_dict:
            continue

        geom = row.geometry
        if geom.is_empty or not isinstance(geom, LineString):
            print(f"Line {i} is empty or not a LineString.")
            continue

        points = intersections_dict[i]
        snapped = [Point(snap_map.get((pt.x, pt.y), (pt.x, pt.y))) for pt in points]

        try:
            splitter = gpd.GeoSeries(snapped).union_all()
            parts = split(geom, splitter)
        except Exception as e:
            print(f"Problem while splitting {i}: {e}")
            continue

        for segment in parts.geoms:
            if segment.geom_type != 'LineString':
                skipped_segments += 1
                continue
        
            if segment.length < 1e-3:
                short_segments += 1
                continue

            coords = list(segment.coords)

            for j in range(len(coords) - 1):
                p1_raw, p2_raw = coords[j], coords[j + 1]
                p1 = tuple(np.round(snap_map.get(p1_raw, p1_raw), 3))
                p2 = tuple(np.round(snap_map.get(p2_raw, p2_raw), 3))

                if p1 == p2:
                    skipped_segments += 1
                    continue

                line_seg = LineString([p1, p2])

                if not line_seg.is_valid:
                    skipped_segments += 1
                    continue

                for pt in [p1, p2]:
                    if pt not in G: # if not a line_intersection, postnl or distribution point then add as waypoint
                        G.add_node(pt, geometry=Point(pt), ntype='waypoint', risk=0)
                        new_waypoints += 1

                edge_key = tuple(sorted((p1, p2)))
                if edge_key in added_edges:
                    continue

                added_edges.add(edge_key)

                G.add_edge(
                    p1, p2,
                    geometry=line_seg,
                    length=line_seg.length,
                    name=row.get("name"),
                    etype=row.get("area_type"),
                    risk=row.get("risk"),
                    height=row.get("Height"),
                )

                edge_count += 1
    
    # Add lines without intersections as single edge
    for i, row in lines.iterrows():
        if i in intersections_dict:
            continue  # already handled

        geom = row.geometry
        if geom.is_empty or not isinstance(geom, LineString):
            continue

        coords = list(geom.coords)
        for j in range(len(coords) - 1):
            p1 = tuple(np.round(coords[j], 3))
            p2 = tuple(np.round(coords[j + 1], 3))

            if p1 == p2:
                continue

            if p1 not in G:
                G.add_node(p1, geometry=Point(p1), ntype='waypoint', risk=0)
            if p2 not in G:
                G.add_node(p2, geometry=Point(p2), ntype='waypoint', risk=0)

            edge_key = tuple(sorted((p1, p2)))
            if edge_key in added_edges:
                continue

            added_edges.add(edge_key)
            G.add_edge(
                p1, p2,
                geometry=LineString([p1, p2]),
                length=LineString([p1, p2]).length,
                name=row.get("name"),
                etype=row.get("area_type"),
                risk=row.get("risk"),
                height=row.get("Height"),
            )

    print(f"\n Graph update summary:")
    print(f" - Lines found: {len(lines)}")
    print(f" - Intersections found: {len(intersections_dict)}")
    print(f" - Line intersection nodes added: {added_nodes}")
    print(f" - Edges added: {edge_count}")
    print(f" - New waypoint nodes added: {new_waypoints}")
    print(f" - Skipped short/invalid segments: {skipped_segments}")
    print(f" - Final graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    return G