import pandas as pd
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def read_data(filename):
    df = pd.read_csv(os.path.join("../clean_data/", filename))
    return df 

def run():
    df_train = read_data("train_cleaned.csv")

    X = df_train.drop("price", axis=1)
    y = df_train["price"]
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(max_depth=10, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_valid)
    
    rmse = mean_squared_error(y_valid, y_pred, squared=False)
    
if __name__ == "__main__":
    run()