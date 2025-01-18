import re
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

def count_population(pop_data: pd.DataFrame):
    
    x_range = pop_data["X"].max() - pop_data["X"].min()
    x_dist = x_range * 111.32 * np.cos(np.mean(pop_data["Y"]) * np.pi / 180)
    
    y_range = pop_data["Y"].max() - pop_data["Y"].min()
    y_dist = y_range * 111.32

    area = x_dist * y_dist
    average_density = pop_data["Z"].mean()
    return area * average_density

coords_re = re.compile(r"(-?\d+\.?\d*),\s*(-?\d+\.?\d*)")
def coords_to_xy(coords):
    if not coords_re.match(coords):
        raise ValueError("Invalid coordinates")
    return reversed([float(coord) for coord in coords.split(",")])

if __name__ == "__main__":
    data = read_data()
    # Colorado: -109, -102, 37, 41
    filtered = filter_data(data, -85.9, -84, 33.9, 34)
    
