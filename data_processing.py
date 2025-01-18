import numpy as np
import pandas as pd

csv_filename = "usa_pd_2020_1km_ASCII_XYZ.csv"

def read_data():
    data = pd.read_csv(csv_filename) # this is the line
    print(data)

read_data()