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
        df = pd.read_csv(Path("clean_data",filename))
        
    return df

def clean_property_type(data):
    data["property_type"] = data["property_type"].apply(lambda x: x.lower())
    
    pattern = re.compile(r'\d{1} room.')
    data["property_type"] = data["property_type"].apply(lambda x: pattern.sub('', x))
    data["property_type"] = data["property_type"].apply(lambda x: x.strip())
    data["property_type"] = data["property_type"].str.replace("land only", "landed")
    data["property_type"] = data["property_type"].str.replace("shophouse", "walk-up")
    
    return data

def fill_missing_values(data):
    # tenure
    lease_hdb = ["hdb", "hdb executive"]
    data["tenure"] = data["tenure"].fillna(pd.Series(np.where((data["property_type"].isin(lease_hdb)), "99-year leasehold", pd.NA )))
    dict_tenure = read_dataframe("dict_tenure.pkl")
    data["tenure"] = data["tenure"].fillna(data["property_type"].map(dict_tenure))
    
    # beds and baths
    dict_bed_bath = read_dataframe("dict_bed_bath.pkl")
    data["num_baths"] = data["num_baths"].fillna(data["property_type"].map(dict_bed_bath[0]))
    data["num_beds"] = data["num_beds"].fillna(data["property_type"].map(dict_bed_bath[1]))
    
    # built_year
    dict_built_year_subzone = read_dataframe("dict_built_year_subzone.pkl")
    data["built_year"] = data["built_year"].fillna(data["subzone"].map(dict_built_year_subzone))
    data["built_year"] = data["built_year"].fillna(2022)

    # total_num_units
    data["total_num_units"] = data["total_num_units"].fillna(-1)
    return data

def update_data(data):
    lat_update = read_dataframe("dict_lat_update.pkl")
    lng_update = read_dataframe("dict_lng_update.pkl")
    subzone_update = read_dataframe("dict_subzone_update.pkl")
    planning_area_update = read_dataframe("dict_planning_area_update.pkl")
    
    for addr in lat_update.keys():
        data["lat"] = np.where(data["address"] == addr, lat_update[addr], data["lat"] )
        data["lng"] = np.where(data["address"] == addr, lng_update[addr], data["lng"] )
    
    data["subzone"] = data["subzone"].fillna(data["address"].map(subzone_update))
    data["planning_area"] = data["planning_area"].fillna(data["address"].map(planning_area_update))
    
    return data

def drop_columns(data):
    # drop unimportant columns
    cols_drop = ["listing_id", "title", "address", "property_name", "available_unit_types", "property_details_url", 
             "elevation", "subzone", "floor_level"]
    data = data.drop(cols_drop, axis=1)
    
    return data
    
def create_features(data):
    
    # bin "tenure"
    data["tenure"] = np.where(data["tenure"].isin(["99-year leasehold", "freehold"]),
                              data["tenure"].str.title(),
                              "Others")
    
    # create mean price for each tenure category
    dict_tenure_prices = read_dataframe("dict_tenure_price.pkl")
    data["mean_tenure_price"] = data["tenure"].map(dict_tenure_prices)
    
    # create "active years" by subtracting built year from current year
    data["active_years"] = 2022 - data["built_year"]
    
    # create "remaining lease" 
    data["lease_year"] = data["tenure"].str.extract('([0-9]{2,3})', expand=False).str.strip()
    data["lease_year"] = data["lease_year"].fillna("-1")
    data["lease_year"] = data["lease_year"].astype(int)
    data["end_lease"] = np.where(data["lease_year"]!=-1, data["built_year"] + data["lease_year"], data["lease_year"])
    data["remaining_lease"] = np.where(data["lease_year"]!=-1, data["end_lease"] - 2022, data["lease_year"])

    data = data.drop(["end_lease", "lease_year"], axis=1)
    
    dict_target_encode = read_dataframe("dict_target_encode.pkl")
    data["planning_area_mean"] = data["planning_area"].map(dict_target_encode)
    data["planning_area_mean"] = data["planning_area_mean"].fillna(-1)
    
    # create "total_rooms" by summing up bedrooms and bathrooms
    data["total_rooms"] = data["num_beds"] + data["num_baths"]
    
    # create "size_per_room"
    data["size_per_room"] = data["size_sqft"]/data["total_rooms"]
    
    # create mean price for each property
    dict_property_price = read_dataframe("dict_property_price.pkl")
    data["mean_property_type"] = data["property_type"].map(dict_property_price)
    
    # calculate distance from nearest mrt
    df_data_mrt_cross = read_dataframe("coord_min_dist_mrt.csv")
    df_min_dist_mrt = df_data_mrt_cross.groupby(["lat_x", "lng_x"])["distance"].min().reset_index()
    data["lat"] = data["lat"].astype(float)
    data["lng"] = data["lng"].astype(float)
    df_min_dist_mrt["distance"] = df_min_dist_mrt["distance"].astype(str)
    df_min_dist_mrt["distance"] = df_min_dist_mrt["distance"].apply(lambda x: x.replace("km" ,""))
    df_min_dist_mrt["distance"] = df_min_dist_mrt["distance"].astype(float)
    data = data.merge(df_min_dist_mrt, left_on=["lat", "lng"], right_on=["lat_x", "lng_x"], how="left")
    data = data.drop(["lat_x", "lng_x"],axis=1)
    data["distance"] = data['distance'].fillna(-1)
    
    # number of shopping malls
    dict_shopping_area = read_dataframe("dict_shopping_area.pkl")
    data["num_shopping_malls"] = data["planning_area"].map(dict_shopping_area)
    data["num_shopping_malls"] = data["num_shopping_malls"].fillna(-1)

    # one hot encode tenure and property type
    def encode_and_bind(original_dataframe, feature_to_encode):
        dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
        res = pd.concat([original_dataframe, dummies], axis=1)
        return(res)

    data = encode_and_bind(data, "tenure")
    data = encode_and_bind(data, "property_type")
    data = encode_and_bind(data, "furnishing")
    
    return data
    
    
    