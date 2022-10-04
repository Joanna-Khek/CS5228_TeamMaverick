import pandas as pd
import re
import numpy as np
import folium
import os
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

def check_shape(data):
    """
    input: DataFrame
    return: None. Prints the shape of the DataFrame to sys.stdout
    """
    num_points, num_attributes = data.shape
    return print("There are {} data points and {} features".format(num_points, num_attributes))


def missing_values_table(df):
    """
    input: DataFrame
    return: DataFrame containing info on missing values for each column
    """
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns

def clean_property_type(data):
    """
    input: DataFrame with column "property_type"
    return: DataFrame with column "property_type" cleaned
    """

    # Normalise casing, remove information concerning number of rooms (e.g. Hdb 4 rooms --> hdb)
    data["property_type"] = data["property_type"].apply(lambda x: x.lower())
    pattern = re.compile(r'\d{1} room.')
    data["property_type"] = data["property_type"].apply(
        lambda x: pattern.sub('', x))
    data["property_type"] = data["property_type"].apply(lambda x: x.strip())

    # Grouping categories
    data["property_type"] = data["property_type"].str.replace(
        "land only", "landed")
    data["property_type"] = data["property_type"].str.replace(
        "shophouse", "walk-up")

    return data

def highlight_top(ax, orientation, highlight_col):
    if orientation == "horizontal":
        patch_h = [patch.get_width() for patch in ax.patches]
    elif orientation == "vertical":
        patch_h = [patch.get_height() for patch in ax.patches]
    
    max_value = max(patch_h)
    
    # Check if there are more than 1 largest value
    idx_tallest = [i for i, j in enumerate(patch_h) if j == max_value]
        
    for idx in idx_tallest:
        ax.patches[idx].set_facecolor(highlight_col)

def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores, figure_name, img_dir):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    save_fig(figure_name, img_dir)
    plt.title("Mutual Information Scores")
        
def save_fig(fig_id, img_dir, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(img_dir, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

        