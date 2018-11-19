import csv
import time
import numpy as np
import json
import argparse




def load_data_file(file_path):
    data = []
    with open(file_path) as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    data = data[1:]
    return data


# def imputate(data):
#



if __name__ == "__main__":
    input_path = "/tmp/m4_monthly.csv"
    output_path = "/tmp/m4_monthly_0.1.json"
    num_vars = 15
    data = load_data_file(input_path)
    imputated = imputate(data)