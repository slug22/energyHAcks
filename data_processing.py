import numpy as np
import pandas as pd

csv_filename = "usa_pd_2020_1km_ASCII_XYZ.csv"

def read_data():
    data = pd.read_csv(csv_filename) # this is the line
    return data

def filter_data(data: pd.DataFrame, x_min, x_max, y_min, y_max):
    return data[
        (data["X"] > x_min) & (data["X"] < x_max) &
        (data["Y"] > y_min) & (data["Y"] < y_max)]

if __name__ == "__main__":
    data = read_data()
    filtered = filter_data(data, -150, -148, 65, 68)
    print(filtered)
