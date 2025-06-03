import geopandas as gpd
import pandas as pd


def read_geojson(file_path, depot, all=True):
    """
    Reads a GeoJSON file

    Args:
        file_path (str): Path to the GeoJSON file.
        depot (str): The depot city to filter distribution points.
        all (bool): If True, return separate dataframes for lines, polygons, postnl points, distribution points and no fly zones.
    """
    gdf = gpd.read_file(file_path)
    gdf = gdf.to_crs(epsg=28992)

    if all: # Get separate dataframes for lines, polygons, postnl points, distribution points and no fly zones
        print("Getting separate dataframes for lines, polygons, postnl points, distribution points and no fly zones")

        lines_gdf = gdf[(gdf.geometry.type == 'LineString') & (gdf['risk'] != 'no_fly_zone')].copy()
        lines_gdf['risk'] = lines_gdf['risk'].astype(float).round(3).fillna(0) # Ensure 'risk' is float and fill NaNs with 0
        if lines_gdf.empty:
            print("No lines found in the GeoJSON file.")

        polygons_gdf = gdf[(gdf.geometry.type == 'Polygon') & (gdf['risk'] != 'no_fly_zone')].copy()
        polygons_gdf['risk'] = polygons_gdf['risk'].astype(float).round(3).fillna(0)  # Ensure 'risk' is float and fill NaNs with 0
        if polygons_gdf.empty:
            print("No polygons found in the GeoJSON file.")

        post_nl_gdf = gdf[gdf['area_type'] == 'postnl point'].copy()
        post_nl_gdf['risk'] = post_nl_gdf['risk'].astype(float).round(3).fillna(0)  # Ensure 'risk' is float and fill NaNs with 0
        if post_nl_gdf.empty:
            print("No PostNL points found in the GeoJSON file.")

        no_fly_zones_gdf = gdf[gdf['risk'] == 'no_fly_zone'].copy()
        no_fly_zones_gdf['area_type'] = 'No-fly zone'  # Ensure area_type is set correctly
        if no_fly_zones_gdf.empty:
            print("No no-fly zones found in the GeoJSON file.")

        print(f"Looking for distribution points in {depot}")
        distribution = get_distribution_points('/Users/cmartens/Documents/thesis_cf_martens/distribution_centers/output/postnl_distribution_cleaned.json', depot)
        print(f"Found {len(distribution)} distribution points in {depot}")
        if distribution.empty:
            print("No distribution points found for the depot.")

        # Set height and risk to 0 for postnl points
        post_nl_gdf['Height'] = 0
        post_nl_gdf['risk'] = 0

        # Add distribution point to the postnl points dataframe
        post_nl_gdf = pd.concat([post_nl_gdf, distribution], ignore_index=True)

        print(f"Found {len(lines_gdf)} lines, {len(polygons_gdf)} polygons, {len(post_nl_gdf)} postnl points, {len(distribution)} distribution point and {len(no_fly_zones_gdf)} no-fly zones")

        return gdf, lines_gdf, polygons_gdf, post_nl_gdf, no_fly_zones_gdf

    # Get only the main dataframe
    print(f"Getting only the main dataframe")

    if gdf.empty:
        print("No data found in the GeoJSON file.")
        return None

    print(f"Found {len(gdf)} geometries in the GeoJSON file.")

    return gdf

def get_distribution_points(file_path, depot):
    """
    Reads a GeoJSON file and returns the distribution points
    for all depot cities in the provided list.

    Args:
        file_path (str): Path to the GeoJSON file.
        depot (list[str]): List of depot cities to filter distribution points.

    Returns:
        GeoDataFrame or None: Filtered distribution points or None if empty.
    """
    gdf = gpd.read_file(file_path)
    gdf = gdf.to_crs(epsg=28992)

    # Filter for multiple depots and combine
    distribution = gdf[gdf['depotCity'].isin(depot)].copy()
    distribution.rename(columns={'type': 'area_type'}, inplace=True)

    if distribution.empty:
        print("No distribution points found in the GeoJSON file.")
        return None

    return distribution