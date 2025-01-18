import csv
import numpy as np

csv_filename = "usa_pd_2020_1km_ASCII_XYZ.csv"

def read_data():
    with open(csv_filename, newline = "", encoding = "utf8") as csv_file:
        # newline = "" recommended by csv library
        csv_reader = csv.reader(csv_file)
        
        i = 0
        for row in csv_reader:
            print(row)
            i += 1
            if i > 100:
                break

        # data = list(csv_reader)

    # pairs = {} # { pair: image }
    # base_letter = ord("A")
    # for row in range(2, len(data)):
    #     for col in range(2, len(data[0])):
    #         if data[row][col] not in ("", "~"):
    #             # generate pair name
    #             label = chr(col - 2 + base_letter) + chr(row - 2 + base_letter)
    #             pairs[label] = data[row][col]

    # keys = list(pairs)
    # print(f"Found {len(keys)} letter pairs")
    # return keys, pairs

read_data()