import pandas as pd
import networkx as nx

def create_graph_with_nodes(gdf):
    """
    Creates a graph with only PostNL and distribution nodes from a GeoDataFrame.

    Args:
        gdf (geopandas.GeoDataFrame): The GeoDataFrame containing the node geometries.
    """
    G = nx.Graph()
    total_rows = len(gdf)
    dist_added = 0
    postnl_added = 0
    skipped = 0

    for index, row in gdf.iterrows():
        if row.get('area_type') == 'distribution':
            if row['geometry'].geom_type == 'Point':
                if pd.isnull(row['id']):
                    row['id'] = index
                G.add_node(int(row['id']), geometry=row['geometry'], ntype='distribution', risk=0)
                dist_added += 1
            else:
                print(f"Skipped distribution {index}: geometry is not Point ({row['geometry'].geom_type})")
                skipped += 1

        elif row.get('area_type') == 'postnl point':
            if row['geometry'].geom_type == 'Point':
                G.add_node(int(index), geometry=row['geometry'], ntype='postnl', risk=0)
                postnl_added += 1
            else:
                print(f"Skipped postnl {index}: geometry is not Point ({row['geometry'].geom_type})")
                skipped += 1

    print(f"Graph creation summary:")
    print(f" - Total rows: {total_rows}")
    print(f" - Distributions added: {dist_added}")
    print(f" - PostNL added: {postnl_added}")
    print(f" - Skipped (non-Point geometry): {skipped}")

    return G
