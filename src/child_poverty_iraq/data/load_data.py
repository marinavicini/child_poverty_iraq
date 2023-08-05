import pandas as pd
import os
import requests
import pickle
import geopandas as gpd

import child_poverty_iraq.utils.constants as c


def get_poverty_adm1():
    filename_pov = "../data/raw/poverty_adm1.csv"
    pov_adm1 = pd.read_csv(filename_pov)
    return pov_adm1


def get_poverty_adm0():
    pov_adm1 = get_poverty_adm1()
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


def get_mosaiks_adm0():
    filename_adm0 = "../data/external/ADM_2_regions_RCF_global_dense_aggregated_to_ADM0_pop_weight=True.p"

    # Load the pickle file
    with open(filename_adm0, "rb") as file:
        mosaiks_adm0 = pickle.load(file)

    mosaiks_adm0.reset_index(inplace=True)

    return mosaiks_adm0


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

    return gdf


def get_geom_adm0():
    return get_geom_cgaz(c.url_tjson_adm0, c.filepath_tjson_adm0)


# def get_geom_adm1():
#     return get_geom_cgaz(c.url_tjson_adm1, c.filepath_tjson_adm1)

# def get_geom_adm2():
#     return get_geom_cgaz(c.url_tjson_adm2, c.filepath_tjson_adm2)
