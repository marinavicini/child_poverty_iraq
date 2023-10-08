import pandas as pd
import os
import requests
import pickle
import geopandas as gpd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import gc

import child_poverty_iraq.utils.constants as c


def get_poverty_feat_adm1(all_dims=False):
    """There are two dataframe with adm1 estimates, the first contains
    all the dimensions but it has less countries"""
    if all_dims:
        filename_pov = "../data/raw/subnational2023_subdomains_poverty.csv"
    else:
        filename_pov = "../data/raw/poverty_adm1.csv"
    pov_adm1 = pd.read_csv(filename_pov)
    return pov_adm1


def get_poverty_adm0():
    pov_adm1 = get_poverty_feat_adm1()
    # Get only variable at national level
    pov_adm0 = pov_adm1[
        [
            "countrycode",
            "survey",
            "survey_round",
            "year_chpov",
            "nat_deprived_sev",
            "nat_deprived_mod",
            "popnational0017",
        ]
    ].drop_duplicates()

    # reset index
    pov_adm0.reset_index(inplace=True, drop=True)
    return pov_adm0


def get_file_names(directory_path):
    # Get a list of all files in the directory
    file_list = [
        os.path.splitext(f)[0]
        for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f))
    ]
    file_list = list(set(file_list))
    return file_list


def get_poverty_geom_adm1():
    directory_path = "../data/external/ADM1_geojson"
    file_list = get_file_names(directory_path)

    geom_pov_adm1 = gpd.GeoDataFrame()

    for cc in tqdm(file_list):
        filepath = f"{directory_path}/{cc}.geojson"
        tmp = gpd.read_file(filepath)
        geom_pov_adm1 = pd.concat([geom_pov_adm1, tmp])

    return geom_pov_adm1


def get_mosaiks(filename):
    filepath = f"../data/external/{filename}.p"

    # Load the pickle file
    with open(filepath, "rb") as file:
        mosaiks = pickle.load(file)

    return mosaiks


def get_mosaiks_adm0():
    mosaiks = get_mosaiks(
        "ADM_2_regions_RCF_global_dense_aggregated_to_ADM0_pop_weight=True"
    )
    mosaiks.reset_index(inplace=True)
    return mosaiks


def get_mosaiks_feat_adm1():
    """'
    Get mosaiks features without geometries
    """
    mosaiks = get_mosaiks(
        "ADM_2_regions_RCF_global_dense_aggregated_to_ADM1_pop_weight=True"
    )
    mosaiks.reset_index(inplace=True)
    return mosaiks


def get_mosaiks_feat_adm2():
    return get_mosaiks("ADM_2_regions_RCF_global_dense_pop_weight=True")


def get_mosaiks_adm1():
    # Mosaiks
    mosaiks_adm1 = get_mosaiks_feat_adm1()
    print(f"Mosaiks shape: {mosaiks_adm1.shape}")

    # Mosaiks Geometry
    geom_mos_adm1 = get_mosaiks_geom_adm1()
    print(f"Geometry shape: {geom_mos_adm1.shape}")

    # Merge
    mosaiks_adm1 = pd.merge(
        mosaiks_adm1,
        geom_mos_adm1,
        how="left",
        left_on="ADM1_shape",
        right_on="shapeID",
    )
    print(mosaiks_adm1.shape)

    # Check missing values in merging
    print(
        f"Number of obs with missing geometry: {mosaiks_adm1[mosaiks_adm1['geometry'].isnull()].shape[0]}"
    )

    # Drop rows with missing geometry
    mosaiks_adm1 = mosaiks_adm1.dropna(subset=["geometry"])
    print(mosaiks_adm1.shape)

    # Transform in geo dataframe
    mosaiks_adm1 = gpd.GeoDataFrame(mosaiks_adm1, geometry="geometry")

    return mosaiks_adm1


def get_geom_cgaz(url, filepath):
    # If the file already exists, no need to download it
    if os.path.exists(filepath):
        print(f"The file '{filepath}' exists.")
    else:
        # Download the topojson file if it is not already present
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful

        with open(filepath, "wb") as file:
            file.write(response.content)

        print(f"File downloaded successfully as {filepath}")

    # Use geopandas to read the TopoJSON file
    gdf = gpd.read_file(filepath)

    # Set the CRS
    gdf.crs = "EPSG:4326"

    return gdf


def get_mosaiks_geom_adm0():
    return get_geom_cgaz(c.url_tjson_adm0, c.filepath_tjson_adm0)


def get_mosaiks_geom_adm1():
    return get_geom_cgaz(c.url_tjson_adm1, c.filepath_tjson_adm1)


def get_mosaiks_geom_adm2():
    return get_geom_cgaz(c.url_tjson_adm2, c.filepath_tjson_adm2)


def get_poverty_adm1(all_dims=False):
    """
    Get poverty features with geometry"""
    # Poverty
    pov_adm1 = get_poverty_feat_adm1(all_dims)
    print(f"Poverty shape: {pov_adm1.shape}")

    # Geometry
    geom_pov_adm1 = get_poverty_geom_adm1()
    print(f"Geometry shape: {geom_pov_adm1.shape}")

    # Merge
    pov_adm1 = pd.merge(
        pov_adm1, geom_pov_adm1, how="left", left_on="geocode", right_on="SUBREGION"
    )
    pov_adm1 = gpd.GeoDataFrame(pov_adm1, geometry="geometry")
    print(f"Poverty + geometry shape: {pov_adm1.shape}")

    return pov_adm1


def make_valid(x):
    if x.is_valid:
        return x
    else:
        return x.buffer(0)


def get_area(x):
    try:
        return x.area
    except:
        return 0


def merge_mosaiks_pov_adm1(pov_adm1, mosaiks_adm1, threshold=0.51):
    # DROP MISSING GEOMETRY IN POV ADM1 (i.e. COL e DOM)
    pov_adm1.dropna(subset=["geometry"], inplace=True)

    # Check CRS
    assert pov_adm1.crs == mosaiks_adm1.crs

    # Valid geometry
    pov_adm1["geometry"] = pov_adm1.apply(lambda x: make_valid(x["geometry"]), axis=1)
    mosaiks_adm1["geometry"] = mosaiks_adm1.apply(
        lambda x: make_valid(x["geometry"]), axis=1
    )

    # Copy geometry
    mosaiks_adm1["geom_mos"] = mosaiks_adm1["geometry"]
    pov_adm1["geom_pov"] = pov_adm1["geometry"]

    # Merge dataframe
    tmp = gpd.sjoin(pov_adm1, mosaiks_adm1, how="left", predicate="intersects")
    print(tmp.shape)

    # Intersection between geometry
    tmp["geom_inter"] = tmp.apply(
        lambda x: x["geom_pov"].intersection(x["geom_mos"]), axis=1
    )

    # Percentage of common areas
    tmp["area_inter"] = tmp.apply(lambda x: get_area(x["geom_inter"]), axis=1)
    tmp["area_pov"] = tmp.apply(lambda x: get_area(x["geom_pov"]), axis=1)
    tmp["perc_pov"] = tmp["area_inter"] / tmp["area_pov"]

    # Keep only geometries above threshold
    merged = tmp[tmp["perc_pov"] > threshold].copy()
    print(merged.shape)

    return merged


def make_dataset_ADM1(all_dims=False):
    # Get mosaiks features
    mosaiks_adm1 = get_mosaiks_adm1()

    # Get poverty data
    pov_adm1 = get_poverty_adm1(all_dims)

    # Merge them
    merged = merge_mosaiks_pov_adm1(pov_adm1, mosaiks_adm1, threshold=0.51)
    return merged


################
# ADM2
################


def adm2_for_country(cc="IRQ"):
    """Get Mosaiks features with geometry for country"""
    # Mosaiks
    filepath = f"../data/interim/{cc.upper()}_mosaiks_ADM2.csv"
    if os.path.exists(filepath):
        print(f"The file '{filepath}' exists.")
        irq_mosaiks_adm2 = pd.read_csv(filepath)
        print(irq_mosaiks_adm2.shape)

    else:
        mosaiks_adm2 = get_mosaiks_feat_adm2()
        mosaiks_adm2["countrycode"] = mosaiks_adm2["shapeID"].apply(lambda x: x[:3])
        irq_mosaiks_adm2 = mosaiks_adm2[mosaiks_adm2["countrycode"] == cc]
        print(irq_mosaiks_adm2.shape)

        del mosaiks_adm2
        gc.collect()

        # Save
        irq_mosaiks_adm2.to_csv(filepath, index=False)

    # Geometry
    filepath_geom = f"../data/interim/{cc}_geom_mosaiks_ADM2.geojson"
    if os.path.exists(filepath_geom):
        irq_geom_mos_adm2 = gpd.read_file(filepath_geom)

    else:
        geom_mos_adm2 = get_mosaiks_geom_adm2()
        irq_geom_mos_adm2 = geom_mos_adm2[geom_mos_adm2["shapeGroup"] == cc]
        del geom_mos_adm2
        gc.collect()

        # A column has the same name as the index
        # Check if 'shapeID' is both an index level and a column label
        if (
            "shapeID" in irq_geom_mos_adm2.columns
            and "shapeID" in irq_geom_mos_adm2.index.names
        ):
            new_index_name = "index"  # Choose a new name for the index
            irq_geom_mos_adm2.index.names = [new_index_name]

        # Save
        irq_geom_mos_adm2[["shapeID", "geometry"]].to_file(
            filepath_geom, driver="GeoJSON"
        )

    print(irq_geom_mos_adm2.shape)

    # Merge
    mosaiks_adm2 = pd.merge(
        irq_mosaiks_adm2,
        irq_geom_mos_adm2,
        how="left",
        left_on="shapeID",
        right_on="shapeID",
    )
    print(mosaiks_adm2.shape)

    # Plot
    mosaiks_adm2 = gpd.GeoDataFrame(mosaiks_adm2)
    mosaiks_adm2.plot()
    plt.show()

    return mosaiks_adm2
