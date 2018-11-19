#!/usr/bin/env python3
from src.OpenForecast import *


def test_train():
    input = dict()
    input['mode'] = "train"
    input['data_path'] = "data://username/collection/dataset_1.0.json"
    input['model_output_path'] = "data://username/collection/model_0.1.0.zip"
    input['training_time'] = 300
    input['model_complexity'] = 0.65
    input['forecast_length'] = 10
    return apply(input)


def test_retrain():
    input = dict()
    input['mode'] = "train"
    input['data_path'] = "data://username/collection/dataset_1.1.json"
    input['model_input_path'] = "data://username/collection/model_0.1.0.zip"
    input['model_output_path'] = "data://username/collection/model_0.1.1.zip"


def test_forecast():
    input = dict()
    input['mode'] = "forecast"
    input['model_input_path'] = "data://username/collection/model_0.1.0.zip"
    input['graph_save_path'] = "data://username/collection/my_api_chart.png"
    input['data_path'] = "data://username/collection/dataset_1.0.json"
    input['forecast_length'] = 30
    input['io_noise'] = 0.05
    print(input)
    return apply(input)

if __name__ == "__main__":
  # result = test_forecast()
  # result = test_retrain()
  result = test_train()
  print(result)
