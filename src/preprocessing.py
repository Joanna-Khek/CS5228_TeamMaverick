import re
import argparse
import numpy as np
import pickle
import pandas as pd
import os
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder


def read_dataframe(filename):
    if filename.endswith(".pkl"):
        with open(Path("clean_data", filename), "rb") as f:
            df = pickle.load(f)
    else:
        df = pd.read_csv(Path("clean_data", filename))

    return df


def clean_property_type(data):
    data["property_type"] = data["property_type"].apply(lambda x: x.lower())

    pattern = re.compile(r'\d{1} room.')
    data["property_type"] = data["property_type"].apply(
        lambda x: pattern.sub('', x))
    data["property_type"] = data["property_type"].apply(lambda x: x.strip())
    data["property_type"] = data["property_type"].str.replace(
        "land only", "landed")
    data["property_type"] = data["property_type"].str.replace(
        "shophouse", "walk-up")

    return data


def fill_missing_values(data):
    # tenure
    lease_hdb = ["hdb", "hdb executive"]
    data["tenure"] = data["tenure"].fillna(pd.Series(
        np.where((data["property_type"].isin(lease_hdb)), "99-year leasehold", pd.NA)))
    dict_tenure = read_dataframe("dict_tenure.pkl")
    data["tenure"] = data["tenure"].fillna(
        data["property_type"].map(dict_tenure))

    # beds and baths
    dict_bed_bath = read_dataframe("dict_bed_bath.pkl")
    data["num_baths"] = data["num_baths"].fillna(
        data["property_type"].map(dict_bed_bath[0]))
    data["num_beds"] = data["num_beds"].fillna(
        data["property_type"].map(dict_bed_bath[1]))

    # built_year
    dict_built_year_subzone = read_dataframe("dict_built_year_subzone.pkl")
    data["built_year"] = data["built_year"].fillna(
        data["subzone"].map(dict_built_year_subzone))
    data["built_year"] = data["built_year"].fillna(2022)

    return data


def update_data(data):
    lat_update = read_dataframe("dict_lat_update.pkl")
    lng_update = read_dataframe("dict_lng_update.pkl")
    subzone_update = read_dataframe("dict_subzone_update.pkl")
    planning_area_update = read_dataframe("dict_planning_area_update.pkl")

    for addr in lat_update.keys():
        data["lat"] = np.where(data["address"] == addr,
                               lat_update[addr], data["lat"])
        data["lng"] = np.where(data["address"] == addr,
                               lng_update[addr], data["lng"])

    data["subzone"] = data["subzone"].fillna(
        data["address"].map(subzone_update))
    data["planning_area"] = data["planning_area"].fillna(
        data["address"].map(planning_area_update))

    return data


def drop_columns(data):
    # drop unimportant columns
    cols_drop = ["listing_id", "title", "address", "property_name", "available_unit_types", "property_details_url",
                 "elevation", "subzone", "furnishing", "total_num_units", "floor_level", 'nearest_mrt_code',
                 'nearest_mall_index',
                 'nearest_commercial_centre_index',
                 'nearest_primary_school_index',
                 'nearest_secondary_school_index']
    data = data.drop(cols_drop, axis=1)

    return data


def create_features(data):
    # bin "tenure"
    data["tenure"] = np.where(data["tenure"].isin(["99-year leasehold", "freehold"]),
                              data["tenure"].str.title(),
                              "Others")

    # create "active years" by subtracting built year from current year
    data["active_years"] = 2022 - data["built_year"]

    # target encode planning area
    dict_target_encode = read_dataframe("dict_target_encode.pkl")
    data["planning_area_mean"] = data["planning_area"].map(dict_target_encode)

    # one hot encode tenure and property type and nearest MRT line
    def encode_and_bind(original_dataframe, feature_to_encode):
        dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
        res = pd.concat([original_dataframe, dummies], axis=1)
        return (res)

    data = encode_and_bind(data, "tenure")
    data = encode_and_bind(data, "property_type")
    data = encode_and_bind(data, "nearest_mrt_line")

    return data
