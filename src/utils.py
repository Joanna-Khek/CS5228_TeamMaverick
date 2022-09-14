import pandas as pd
import re


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
