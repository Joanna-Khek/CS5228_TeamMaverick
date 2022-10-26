import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import save_fig
from src.preprocessing import encode_and_bind

def merge_mrt_info(df_mrt, df_raw, df_aux, df_train_eda):
    
    df_clean = df_raw.copy()
    
    # create a mrt code name dictionary for mapping
    dict_mrt = df_mrt.loc[:,["code", "name"]].set_index("code")['name'].to_dict()
    
    # combine information of main data and mrt data auxiliary file
    df_clean_full = pd.concat([df_clean, df_aux], axis=1)
    
    # map nearest mrt to name
    df_clean_full["nearest_mrt"] = df_clean_full["nearest_mrt_code"].map(dict_mrt)
    
    # use only the eda cleaned data as we have removed some dirty data before
    df_train_eda_listing = df_train_eda["listing_id"]
    df_clean_full_listing = df_clean_full[df_clean_full["listing_id"].isin(df_train_eda_listing)]
    
    # add the nearest mrt column to the main df_train data
    add_cols = df_clean_full_listing.loc[:,['nearest_mrt']]
    df_train_eda_full = pd.concat([df_train_eda.reset_index(drop=True), add_cols.reset_index(drop=True)], axis=1)
    
    return df_train_eda_full

def cat_sales_type(df_train_eda_full):
    # categorizing sales type
    df_train_eda_full["sales_type"] = np.where(df_train_eda_full["built_year"] > 2022, "New", "Resale")
    return df_train_eda_full

def cat_price(df_train_eda_full):
    # categorizing price category
    max_price = df_train_eda_full["price"].max()
    bins = [0,300000, 500000, 700000, 900000, max_price]
    labels = ["Below 300K", "300K-500K", "500K-700K", "700K-900K", "Above 900K"]
    df_train_eda_full["price_cat"] = pd.cut(df_train_eda_full["price"], bins=bins, labels=labels)
    return df_train_eda_full

def cat_size(df_train_eda_full):
    max_size = df_train_eda_full["size_sqft"].max()
    bins = [0,2000, 5000, 8000, 10000, max_size]
    labels = ["Below 2000", "2000-5000", "5000-8000", "8000-10000", "Above 10000"]
    df_train_eda_full["size_cat"] = pd.cut(df_train_eda_full["size_sqft"], bins=bins, labels=labels)
    return df_train_eda_full

def cat_lease(df_train_eda_full):
    df_train_eda_full["remaining_lease"] = np.where(df_train_eda_full["property_type"]=="hdb", df_train_eda_full["built_year"]+99-2022, 999)
    bins = [0, 60, 70, 80, 90, 99, 999]
    labels = ["Below 60", "60-70", "70-80", "80-90", "90-99", "Non-HDB"]
    df_train_eda_full["lease_cat"] = pd.cut(df_train_eda_full["remaining_lease"], bins=bins, labels=labels)
    return df_train_eda_full

def plot_clusters(kmeans, data, ylabel, xlabel, img_dir):
    plt.figure(figsize=(12,8))
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
        
    # Plot all the data points a color-code them w.r.t. to their cluster label/id
    plt.scatter(data.iloc[:,0], data.iloc[:,1], c=kmeans.labels_, s=50, cmap=plt.cm.tab10)
    
    # Plot the cluster centroids as fat plus signs
    for cluster_id, centroid in enumerate(kmeans.cluster_centers_):
        plt.scatter(centroid[0], centroid[1], marker='+', color='k', s=250, lw=5)

    plt.tight_layout()
    save_fig("kmeans_location", img_dir)
    plt.show()
    
def add_coordinate(data, new_lat, new_lng):
    data = data.append({'lat': new_lat,
                        'lng': new_lng},
                        ignore_index=True)
    return data

def filtered_location(data, mrt_data, preferred_location):
    """
    This function takes in user's preferred location and returns nearby listings
    """
    # lowercase station name
    mrt_data["station_name"] =  mrt_data["station_name"].apply(lambda x: x.lower())
    
    # lowecase the preferred location name
    preferred_location = preferred_location.lower()
    
    # lat lng data
    df_lat_lng = data.loc[:,["lat", "lng"]]
    
    # Check if input location is valid
    pref_locat = mrt_data[mrt_data["station_name"] == preferred_location]
    if len(pref_locat) == 0:
        raise Exception("Input location is not a valid station name")
    else:
        # grab the coordinate of the preferred location and add them to the df_lat_lng dataframe
        pref_locat_coord_lat = pref_locat["lat"].values[0]
        pref_locat_coord_lng = pref_locat["lng"].values[0]
        df_lat_lng_full = add_coordinate(df_lat_lng, pref_locat_coord_lat, pref_locat_coord_lng)
        
        # perform kmeans clustering. 
        kmeans = KMeans(n_clusters=10, random_state=0).fit(df_lat_lng_full)
        df_lat_lng_full['cluster'] = kmeans.labels_
        df_lat_lng["cluster"] = df_lat_lng_full["cluster"].iloc[:-1]
        data["cluster"] = df_lat_lng["cluster"]
        
        # Locate the cluster corresponding to the preferred_location and filter all the listings belonging to same cluster.
        new_point_cluster = df_lat_lng_full["cluster"].iloc[-1]
        df_location = data[data["cluster"] == new_point_cluster].reset_index().drop("index", axis=1)
        return data, df_location, new_point_cluster
    
    
def preference(pref_sales_type, pref_price_cat, pref_size_cat,
               pref_num_beds, pref_num_baths, pref_lease_cat, new_point_cluster):
    # put preference into a dataframe
    data = {'listing_id': "preference",
            'sales_type': pref_sales_type,
            'price_cat' : pref_price_cat,
            'size_cat': pref_size_cat,
            'num_beds': pref_num_beds,
            'num_baths': pref_num_baths,
            'lease_cat': pref_lease_cat,
            'cluster': new_point_cluster}
    
    data = pd.DataFrame([data])
    return data

def encode(df_main, df_preference):
    
    """"
    One hot encode the categorical features to 
    prepare for computation of cosine similarity
    """
    df_main_clean = df_main.loc[:,["listing_id","sales_type", "price_cat", "size_cat", "num_beds", "num_baths", "lease_cat", "cluster"]]

    # append user's preference
    df_full = df_main_clean.append(df_preference)
    
    # one hot encode
    df_full = encode_and_bind(df_full, "sales_type")
    df_full = encode_and_bind(df_full, "price_cat")
    df_full = encode_and_bind(df_full, "size_cat")
    df_full = encode_and_bind(df_full, "num_beds")
    df_full = encode_and_bind(df_full, "num_baths")
    df_full = encode_and_bind(df_full, "lease_cat")
    cluster_encode = pd.get_dummies(df_full["cluster"]).reindex(columns=range(0, 10), fill_value=0).add_prefix("cluster")
    df_full = pd.concat([df_full, cluster_encode], axis=1)
    df_full = df_full.drop("cluster", axis=1)
    return df_full

def similarity_matrix(df_encode):
    """
    Perform cosine similarity
    """
    df_encode = df_encode.drop(["listing_id"], axis=1)
    similarity_matrix = cosine_similarity(df_encode)
    return similarity_matrix

def identify_top_listings(df_main, sim_matrix, num_recommend, num_pref_row):
    """"
    Loop through all the user's preference (either inputted or based on browsing history) 
    and identify the top_num listings
    """
    avg_sim_score = []
    top_listings = pd.DataFrame()
    
    # loop through all preferences
    for i in range(1, num_pref_row+1):
        pref_row_matrix = sim_matrix[-i]
        # remove last index as it is the preference row sim score which we do not need
        sim_score = pref_row_matrix[:-num_pref_row]
        top_idx = np.argpartition(sim_score,-num_recommend)[-num_recommend:]
        avg_sim_score.append(sim_score)
        top_listings = top_listings.append(df_main[df_main.index.isin(top_idx)])
     
    # store the average similarity score   
    avg_sim_score = np.array(avg_sim_score).mean(axis=0)
    df_main["sim_score"] = avg_sim_score
    df_top_listings = df_main[df_main["listing_id"].isin(top_listings["listing_id"])]
    
    return df_top_listings

def store_browsing_history(browsed_listings, df_pref):
    """"
    To simulate browsed listings, we randomly select one number
    from the listings that we show to our users
    """
    browsed = np.random.randint(df_pref.shape[0], size=1)
    browsed_listings.append(df_pref.iloc[browsed]["listing_id"].values[0])
    return browsed_listings


def browsing_history_recommendation(browsed_listings, df_main, top_recommend_num):
    # check the user's browsed history
    num_prefs = len(browsed_listings)
    df_browsed = df_main[df_main["listing_id"].isin(browsed_listings)]
    df_browsed = df_browsed.loc[:,["listing_id","sales_type", "price_cat", "size_cat", "num_beds", "num_baths", "lease_cat", "cluster"]]
    df_browsed_recommend = encode(df_main, df_browsed)
    sim_matrix = similarity_matrix(df_browsed_recommend)
    df_top_listings = identify_top_listings(df_main, sim_matrix, top_recommend_num, num_prefs)
    df_top_listings = df_top_listings.sort_values(by="sim_score", ascending=False)
    return df_top_listings
    