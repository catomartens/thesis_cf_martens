import folium
import pyproj
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import transform
from shapely.wkt import loads as wkt_loads
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.lines import Line2D
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.patches import Patch

def plot_graph_on_folium(G, plot_all_lines=True, connected_paths=False, gdf_polygons=None, plot_waypoints=True):
    """ Plot a NetworkX graph on a Folium map.
    Args:
        G (networkx.Graph): The graph to plot.
        connected_paths (list): List of tuples containing connected paths.
        gdf_polygons (geopandas.GeoDataFrame): Polygons to plot on the map.
        plot_waypoints (bool): Whether to plot waypoints or not.

    Returns:
        folium.Map: A Folium map with the graph plotted on it.
    """
    
    # Projection from EPSG:28992 to EPSG:4326
    project = pyproj.Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True).transform

    # Find the center of the map
    points_latlon = []
    for node_id, data in G.nodes(data=True):
        geom = data.get('geometry')
        if isinstance(geom, str):
            try:
                geom = wkt_loads(geom)
            except:
                continue
        if isinstance(geom, Point):
            geom_latlon = transform(project, geom)
            points_latlon.append((geom_latlon.y, geom_latlon.x))

    if points_latlon:
        avg_lat = sum(lat for lat, lon in points_latlon) / len(points_latlon)
        avg_lon = sum(lon for lat, lon in points_latlon) / len(points_latlon)
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13, tiles='cartodbpositron')
    else:
        m = folium.Map(location=[52.37, 4.90], zoom_start=13, tiles='cartodbpositron')

    # Add polygons to the map if provided
    if gdf_polygons is not None and not gdf_polygons.empty:
        print(f"Plotting {len(gdf_polygons)} polygons on the map.")

        area_color_map = {
            'Pedestrian and cycling paths': '#4DAF9F',     # dark turquoise – accessible
            'Bridges': '#807DBA',                          # purple-blue – infrastructure
            'Wind turbines': '#A6761D',                    # brown – energy-related
            'High infrastructures': '#756BB1',             # dark purple – prominent, tall
            'Power plants': '#D95F02',                     # vivid orange – high-risk
            'Agricultural lands': '#7BBF6A',
            'Industrial zones': '#1F78B4',                 # steel blue – industrial
            'Commercial zones': '#FDB863',                 # amber – busy urban areas
            'Residential areas': '#D9B900',                # dark yellow – housing
            'Recreational zones': '#33A02C',               # green – leisure
            'Parks': '#A6D96A',                            # light green – nature
            'Retail zones': '#FF7F00',                     # orange – shopping
            'Meadows and open grass': '#66A984',           # moss green – open space
            'Forests and woodlands': '#006D2C',            # dark green – dense vegetation
            'Wetlands': '#4F93C0',                         # steel blue – wet areas
            'Lakes and ponds': '#1F78B4',                  # blue – water bodies
            'Schools and universities': '#D4AF37',         # bronze – educational
            'Hospitals': '#E41A1C',                        # red – urgency
            'Cultural sites': '#984EA3',                   # purple – cultural
            'Religious sites': '#E07B12',                  # burnt orange – formal, symbolic
            'Cemeteries': '#FF7F00',                       # orange – memorial
            'Prisons': '#636363',                          # grey – restricted, institutional
            'No-fly zone': '#B22222',                      # fire red – prohibited area
            'Tracks and rural access roads': '#A0522D',  # saddle brown – logisch voor zandwegen/plattelandswegen
            'Regional roads': '#E6550D',                # oranje-bruin – net wat subtieler dan felrood, past goed tussen wegenkleuren
        }

        for idx, row in gdf_polygons.iterrows():
            poly = row["geometry"]
            area_type = row.get("area_type", "Onbekend")

            if poly.is_valid and poly.geom_type == "Polygon":
                poly_latlon = transform(project, poly)
                coords = [(pt[1], pt[0]) for pt in poly_latlon.exterior.coords]

                # Set color to black if area_type is not in the color map
                if area_type not in area_color_map:
                    base_color = '#000000'

                fill_color = area_color_map[area_type]
                folium.Polygon(
                    locations=coords,
                    color=fill_color,
                    weight=1,
                    opacity=0.7,
                    fill=True,
                    fill_opacity=0.15,
                    fill_color=fill_color,
                ).add_to(m)

    # Plot distribution- en PostNL-points
    for node_id, data in G.nodes(data=True):
        ntype = data.get('ntype')
        if ntype not in ['postnl', 'distribution']:
            continue

        geom = data.get('geometry')
        if isinstance(geom, str):
            try: geom = wkt_loads(geom)
            except: continue
        if not isinstance(geom, Point):
            continue

        geom_latlon = transform(project, geom)

        folium.CircleMarker(
            location=[geom_latlon.y, geom_latlon.x],
            radius=4,
            color='green' if ntype == 'distribution' else 'blue',
            fill=True,
            fill_opacity=0.8,
            popup=f"Node ID: {node_id} ({ntype})"
        ).add_to(m)

    # Plot waypoints, intersections and grid points if requested
    if plot_waypoints:
        for node_id, data in G.nodes(data=True):
            ntype = data.get('ntype')
            if ntype not in ['waypoint', 'line_intersection', 'poly_intersection', 'grid_point', 'polygon_boundary']:
                continue

            geom = data.get('geometry')
            if isinstance(geom, str):
                try:
                    geom = wkt_loads(geom)
                except:
                    continue
            if not isinstance(geom, Point):
                continue

            geom_latlon = transform(project, geom)

            # Get the color based on the node type
            color_map = (
                'orange' if ntype == 'waypoint' else
                'purple' if ntype == 'line_intersection' else
                'red' if ntype == 'poly_intersection' else
                'green' if ntype == 'grid_point' else
                'lightred'
            )
            folium.CircleMarker(
                location=[geom_latlon.y, geom_latlon.x],
                radius=1.5,
                color=color_map,
                fill=True,
                fill_opacity=0.9,
                weight=1,
                popup=f"({ntype}, coord: {geom_latlon.x:.3f}, {geom_latlon.y:.3f})"
            ).add_to(m)

    # Plot all edges
    if plot_all_lines:
        for u, v, data in G.edges(data=True):
            geom = data.get('geometry')
            if isinstance(geom, str):
                try: geom = wkt_loads(geom)
                except: continue
            if not isinstance(geom, LineString):
                try:
                    p1 = G.nodes[u]['geometry']
                    p2 = G.nodes[v]['geometry']
                    if isinstance(p1, Point) and isinstance(p2, Point):
                        geom = LineString([p1, p2])
                    else:
                        continue
                except:
                    continue

            geom_latlon = transform(project, geom)
            coords = [(pt[1], pt[0]) for pt in geom_latlon.coords]
            folium.PolyLine(
                coords,
                color='#ffaa00' if data.get('etype') == 'connector' else '#000000',
                weight=2,
                opacity=0.5,
                popup=f"{data.get('etype', '')} ({data.get('length', 0):.1f} m) Risk: {data.get('risk', 'N/A')}"
            ).add_to(m)

    # Plot paths between distribution and PostNL points
    if connected_paths:
        for dist_id, postnl_id, total_length, path_nodes, path_edges, etypes in connected_paths:
            for geom, u, v in path_edges:
                if isinstance(geom, str):
                    try: geom = wkt_loads(geom)
                    except: continue
                if not isinstance(geom, LineString):
                    continue

                geom_latlon = transform(project, geom)
                coords = [(pt[1], pt[0]) for pt in geom_latlon.coords]
                folium.PolyLine(
                    coords,
                    color='darkred',
                    weight=4,
                    opacity=0.8,
                    popup=f"{dist_id} → {postnl_id} ({total_length:.1f} m)"
                ).add_to(m)

    return m

import folium
import pyproj
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import transform
from shapely.wkt import loads as wkt_loads
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.lines import Line2D

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import contextily as ctx

def plot_uav_segments_with_legend(connected_paths, title=None, postnl_points=None):
    """
    Plot a map of UAV corridor segments colored by edge type (etype), including optional PostNL and distribution points.

    Args:
        connected_paths (list): List of tuples containing route data, including path geometries and etype arrays.
        title (str): Plot title.
        postnl_points (GeoDataFrame): Optional. Points with 'area_type' column ('distribution' or 'postnl point').

    Returns:
        matplotlib.figure.Figure: The generated plot figure.
    """

    # --- Color map for etypes
    area_color_map = {
        # Infrastructuur
        'Motorways and major roads': '#E31A1C',
        'Regional roads': '#D95F02',
        'Tracks and rural access roads': '#B5651D',
        'Living and residential streets': '#FDD835',
        'Pedestrian and cycling paths': '#8FBC8F',
        'Railways': '#4B4B4B',
        'Bridges': '#AAAAAA',
        'Helipads': '#F0027F',
        'Airports and airfields': '#A6CEE3',

        # Energie & communicatie
        'Power lines': '#999999',
        'Power plants': '#FF8C00',
        'Wind turbines': '#C0C0C0',
        'Communication towers': '#FB9A99',
        'High infrastructures': '#888888',

        # Gebouwd gebied
        'Industrial zones': '#4682B4',
        'Commercial zones': '#FFD700',
        'Retail zones': '#FFA500',
        'Residential areas': '#FFEA00',
        'Schools and universities': '#DAA520',
        'Hospitals': '#DC143C',
        'Care homes': '#FF9999',
        'Prisons': '#696969',
        'Religious sites': '#C71585',
        'Cemeteries': '#556B2F',
        'Cultural sites': '#8A2BE2',

        # Natuur
        'Parks': '#7CFC00',
        'Recreational zones': '#20B2AA',
        'Meadows and open grass': '#9ACD32',
        'Forests and woodlands': '#228B22',
        'Agricultural lands': '#ADDE87',
        'Wetlands': '#87CEEB',
        'Lakes and ponds': '#4682B4',
        'Water reservoirs': '#1E90FF',
        'Rivers, canals and streams': '#5DADE2',
        'Natura2000 areas': '#005824',

        # Overig
        'connector': '#D3D3D3',
        'No-fly zone': '#B22222'
    }

    # --- Extract geometries and etypes
    segments, etypes = [], []
    for _, _, _, _, path_geometries, etype_array in connected_paths:
        for (line_tuple, *_), etype in zip(path_geometries, etype_array):
            if isinstance(line_tuple, LineString):
                segments.append(line_tuple)
                etypes.append(etype)

    gdf_segments = gpd.GeoDataFrame({"etype": etypes}, geometry=segments, crs="EPSG:28992")
    print("✅ Loaded segments:", len(gdf_segments))
    gdf_web = gdf_segments.to_crs(epsg=3857)
    colors = gdf_web["etype"].map(area_color_map).fillna("#cccccc")

    # --- Base map
    fig, ax = plt.subplots(figsize=(20, 23), dpi=300)
    gdf_web.plot(color=colors, ax=ax, linewidth=6, zorder=2)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.3, zorder=1)
    ax.set_axis_off()
    ax.set_title(title, fontsize=24)

    # --- Plot PostNL & distribution points
    legend_elements = [
        Line2D([0], [0], color=color, lw=5, label=etype)
        for etype, color in area_color_map.items()
        if etype in gdf_web["etype"].unique()
    ]

    if postnl_points is not None and not postnl_points.empty:
        postnl_points_web = postnl_points.to_crs(epsg=3857)

        if 'area_type' in postnl_points.columns:
            for area_type, group in postnl_points_web.groupby("area_type"):
                if area_type == "distribution":
                    ax.scatter(
                        group.geometry.x, group.geometry.y,
                        marker='o', s=150, c='green', edgecolors='black',
                        label='Distribution point', zorder=6
                    )
                elif area_type == "postnl point":
                    ax.scatter(
                        group.geometry.x, group.geometry.y,
                        marker='o', s=100, c='blue', edgecolors='black',
                        label='PostNL point', zorder=6
                    )
                else:
                    ax.scatter(
                        group.geometry.x, group.geometry.y,
                        marker='x', s=80, c='grey', edgecolors='black',
                        label=f'Other: {area_type}', zorder=6
                    )
        else:
            ax.scatter(
                postnl_points_web.geometry.x, postnl_points_web.geometry.y,
                marker='o', s=100, c='blue', edgecolors='black',
                label='PostNL points', zorder=6
            )

        # Add to legend
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                                      markeredgecolor='black', markersize=10, label='Distribution point'))
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                                      markeredgecolor='black', markersize=10, label='PostNL point'))

    ax.legend(
        handles=legend_elements,
        title="Etype",
        loc="upper left",
        bbox_to_anchor=(1.15, 1),
        title_fontsize=20,
        fontsize=18
    )

    plt.tight_layout()
    return fig


def plot_polygons_static(gdf_polygons, city_name="Unknown",
                         no_fly_zones_gdf=None, postnl_points=None, title=None):
    """
    Plot area-type polygons with consistent color coding on a static background map,
    including optional overlays for no-fly zones and PostNL/distribution points.

    Args:
        gdf_polygons (GeoDataFrame): Polygons with an 'area_type' column and EPSG:28992 CRS.
        city_name (str): Name of the city to use in the plot title.
        no_fly_zones_gdf (GeoDataFrame): Optional. Polygons representing no-fly zones.
        postnl_points (GeoDataFrame): Optional. Points with 'ntype' column: 'postnl' or 'distribution'.

    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure.
    """

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import contextily as ctx

    area_color_map = {
        # Infrastructuur
        'Motorways and major roads': '#E31A1C',
        'Regional roads': '#D95F02',
        'Tracks and rural access roads': '#B5651D',
        'Living and residential streets': '#FDD835',
        'Pedestrian and cycling paths': '#8FBC8F',
        'Railways': '#4B4B4B',
        'Helipads': '#F0027F',
        'Airports and airfields': '#A6CEE3',

        # Energie & communicatie
        'Power lines': '#999999',
        'Power plants': '#FF8C00',
        'Wind turbines': '#C0C0C0',
        'Communication towers': '#FB9A99',
        'High infrastructures': '#888888',

        # Gebouwd gebied
        'Industrial zones': '#4682B4',
        'Commercial zones': '#FFD700',
        'Retail zones': '#FFA500',
        'Residential areas': '#F7B733',
        'Schools and universities': '#DAA520',
        'Hospitals': '#DC143C',
        'Care homes': '#FF9999',
        'Prisons': '#696969',
        'Religious sites': '#C71585',
        'Cemeteries': '#556B2F',
        'Cultural sites': '#8A2BE2',

        # Natuur
        'Parks': '#7CFC00',
        'Recreational zones': '#20B2AA',
        'Meadows and open grass': '#9ACD32',
        'Forests and woodlands': '#228B22',
        'Agricultural lands': '#ADDE87',
        'Wetlands': '#87CEEB',
        'Lakes and ponds': '#4682B4',
        'Water reservoirs': '#1E90FF',
        'Rivers, canals and streams': '#5DADE2',
        'Natura2000 areas': '#005824',

        # Overig
        'connector': '#D3D3D3',
        'No-fly zone': '#B22222'
    }
    if gdf_polygons.empty:
        raise ValueError("GeoDataFrame is empty – no polygons to plot.")

    # Project to Web Mercator
    gdf_web = gdf_polygons.to_crs(epsg=3857)
    gdf_web["color"] = gdf_web["area_type"].map(area_color_map).fillna("#cccccc")

    if no_fly_zones_gdf is not None:
        no_fly_zones_web = no_fly_zones_gdf.to_crs(epsg=3857)

    if postnl_points is not None:
        postnl_points_web = postnl_points.to_crs(epsg=3857)

    # Plot setup
    fig, ax = plt.subplots(figsize=(14, 12))
    gdf_web.plot(ax=ax, facecolor=gdf_web["color"], edgecolor='black', alpha=0.3, linewidth=0.5)

    if no_fly_zones_gdf is not None:
        no_fly_zones_web.plot(ax=ax, facecolor="#B22222", edgecolor='black', alpha=0.4, linewidth=0.7)

    # Plot PostNL & distribution points with scatter for better control
    if postnl_points is not None:
        if 'area_type' in postnl_points.columns:
            for area_type, group in postnl_points_web.groupby("area_type"):
                if area_type == "distribution":
                    ax.scatter(
                        group.geometry.x, group.geometry.y,
                        marker='o', s=150, c='green', edgecolors='black',
                        label='Distribution point', zorder=6
                    )
                elif area_type == "postnl point":
                    ax.scatter(
                        group.geometry.x, group.geometry.y,
                        marker='o', s=100, c='blue', edgecolors='black',
                        label='PostNL point', zorder=6
                    )
                else:
                    ax.scatter(
                        group.geometry.x, group.geometry.y,
                        marker='x', s=80, c='grey', edgecolors='black',
                        label=f'Other: {area_type}', zorder=6
                    )
        else:
            ax.scatter(
                postnl_points_web.geometry.x, postnl_points_web.geometry.y,
                marker='o', s=100, c='blue', edgecolors='black',
                label='PostNL points', zorder=6
            )

    # Basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.6)

    # Title and formatting
    ax.set_title(f"{title}", fontsize=16)
    ax.set_axis_off()

    # Legend
    used_types = gdf_web["area_type"].unique()
    legend_patches = [
        Patch(facecolor=color, edgecolor='black', label=atype)
        for atype, color in area_color_map.items()
        if atype in used_types
    ]

    if postnl_points is not None:
        legend_patches.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                                     markeredgecolor='black', markersize=10, label='Distribution point'))
        legend_patches.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                                     markeredgecolor='black', markersize=10, label='PostNL point'))

    if no_fly_zones_gdf is not None and not no_fly_zones_gdf.empty:
        legend_patches.append(Patch(facecolor='#B22222', edgecolor='black', label='No-fly zone'))

    ax.legend(
        handles=legend_patches,
        title="Area Type",
        loc="upper left",
        bbox_to_anchor=(1.15, 1)
    )

    plt.tight_layout()
    return fig


def plot_uav_corridors(connected_paths, output_path=False, color='red', linewidth=2):
    """
    Generate a static map of UAV corridors using LineStrings from connected paths, overlaid on an OSM basemap.

    Args:
        connected_paths (list): List of tuples (dist_node, postnl_node, total_length, path_nodes, path_edges, etype_array).
        output_path (str): File path to save the generated PDF map.
        color (str): Line color for the UAV corridors.
        linewidth (int): Line width for the UAV corridors.

    Returns:
        GeoDataFrame: GeoDataFrame of the LineStrings used in the plot (in EPSG:3857).
    """

    # Extract valid coordinate paths
    lines = []
    for _, _, _, path_nodes, _, _ in connected_paths:
        coords = [pt for pt in path_nodes if isinstance(pt, tuple) and len(pt) == 2]
        if len(coords) >= 2:
            lines.append(LineString(coords))

    if not lines:
        raise ValueError("No valid corridors found in input paths.")

    # Create GeoDataFrame and reproject
    gdf_lines = gpd.GeoDataFrame(geometry=lines, crs="EPSG:28992")
    gdf_web = gdf_lines.to_crs(epsg=3857)

    # Plot
    ax = gdf_web.plot(figsize=(10, 10), linewidth=linewidth, edgecolor=color)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_axis_off()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Map saved to {output_path}")
