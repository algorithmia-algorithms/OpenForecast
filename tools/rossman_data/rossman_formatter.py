import csv
import datetime
import time
import numpy as np
import json
import argparse

def get_data_for_store(data, store_num):
    r"""
    This function returns the csv data for a single store, filtering out all other stores:
    input:
    data - the rossman csv data loaded by 'load_rossman_data'
    store_num - an integer denoting the store id

    output:
    data - a json object containing all output sequences for the store
    """
    found_data = []
    for row in data:
        id = int(row[0])
        if (id == store_num):
            dayOfTheWeek = int(row[1])
            date = datetime.datetime.strptime(str(row[2]), "%Y-%m-%d")
            unix_timestamp = float(time.mktime(date.timetuple()))
            sales = int(row[3])
            customers = int(row[4])
            open_state = int(row[5])
            promo_day = int(row[6])
            # federal holidays can be a few different varieties, so we need to encode that into fed_holiday variable using the encoding scheme
            fed_holiday = encode_holiday_type(row[7])
            bnk_holiday = int(row[8])
            obj = {'date': unix_timestamp,
                   'sales': sales,
                   'num_of_customers': customers,
                   'open_state': open_state,
                   'day_of_the_week': dayOfTheWeek,
                   'fed_holiday': fed_holiday,
                   'promo_day': promo_day,
                   'bnk_holiday': bnk_holiday}
            found_data.append(obj)
    return found_data


def encode_holiday_type(value):
    if value == "0":
        output = 0
    elif value == "a":
        output = 1
    elif value == "b":
        output = 2
    else:
        output = 3
    return output

def load_data_file(file_path):
    data = []
    with open(file_path) as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    data = data[1:]
    return data


def format_for_algorithm(file_path, num_stores):
    r"""This function restructures the csv data into a format available for a generative timeseries algorithm.
    Some of the rossman store data is of different sequence lengths, to avoid issues in training, we skip over stores that don't have the
    standard '66' days of data that most stores produce.

    Input:
    file_path - local file path to the raw rossmans training data csv
    num_stores - the number of stores you want to use in your dataset

    Output:
    A properly formatted json object containing the serializable tensor, headers and important columns.
    """
    raw_data = load_data_file(file_path)
    store_tensors = []
    cols_to_forecast = []
    headers = []
    stored_stores = 0
    store_itr = 0
    while len(store_tensors) < num_stores:
        store_itr += 1
        store_data = get_data_for_store(raw_data, store_itr+1)
        if len(store_data) == 66:
            cols_to_forecast.append(stored_stores*7)
            headers.append('sales for store #{}'.format(str(store_itr)))
            store_sales = np.asarray([day['sales'] for day in store_data])
            store_customers = np.asarray([day['num_of_customers'] for day in store_data])
            store_open_state = np.asarray([day['open_state'] for day in store_data])
            store_dotw = np.asarray([day['day_of_the_week'] for day in store_data])
            store_fed_h = np.asarray([day['fed_holiday'] for day in store_data])
            store_promo_day = np.asarray([day['promo_day'] for day in store_data])
            store_bnk_h = np.asarray([day['bnk_holiday'] for day in store_data])
            store_tensor = np.stack((store_sales, store_customers, store_open_state, store_dotw, store_fed_h, store_bnk_h, store_promo_day), axis=1)
            store_tensors.append(store_tensor)
            stored_stores += 1
    store_tensors = np.concatenate(store_tensors, axis=1)
    serializable_tensor = store_tensors.tolist()
    output = {'headers': headers, 'tensor': serializable_tensor, 'feature_columns': cols_to_forecast}
    return output


def serialize_to_file(path, object):
    with open(path, 'w') as f:
        json.dump(object, f)
    return "done"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The rossman sales data formatter.")
    parser.add_argument('input_path', type=str, help="The local system path to the rossman training data.")
    parser.add_argument('output_path', type=str, help="The local system path to where the formatted data should live.")
    parser.add_argument('num_of_stores', type=int, help="The number of stores to consolidate into the dataset.")
    args = parser.parse_args()
    result = format_for_algorithm(args.input_path, args.num_of_stores)
    serialize_to_file(args.output_path, result)
