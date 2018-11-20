#!/usr/bin/env python3
from src.OpenForecast import *
import os

def test_train():
    input = dict()
    input['mode'] = "train"
    input['data_path'] = "data://TimeSeries/GenerativeForecasting/formatted_data_rossman_10.json"
    input['model_output_path'] = "local:///tmp/model_0.1.0.zip"
    input['training_time'] = 10
    input['model_complexity'] = 0.65
    input['forecast_length'] = 10
    result = apply(input)

    assert result['final_error'] <= 0.10
    assert len(result['forecast']['sales for store #1']) == 10
    assert os.path.isfile(result['model_output_path'])


def test_retrain():
    input = dict()
    input['mode'] = "train"
    input['data_path'] = "data://TimeSeries/GenerativeForecasting/formatted_data_rossman_10.json"
    input['model_input_path'] = "local:///tmp/model_0.1.0.zip"
    input['model_output_path'] = "local:///tmp/model_0.1.1.zip"
    result = apply(input)

    assert result['final_error'] <= 0.10
    assert len(result['forecast']['sales for store #1']) == 10
    assert os.path.isfile(result['model_output_path'])


def test_forecast():
    input = dict()
    input['mode'] = "forecast"
    input['model_input_path'] = "local:///tmp/model_0.1.0.zip"
    input['graph_save_path'] = "local:///tmp/my_graph.png"
    input['data_path'] = "data://TimeSeries/GenerativeForecasting/formatted_data_rossman_10.json"
    input['io_noise'] = 0.05
    result = apply(input)

    assert result['forecast']['sales for store #9'][-1] >= 4000
    assert result['forecast']['sales for store #9'][-1] <= 5100
    assert os.path.isfile(result['graph_save_path'])

if __name__ == "__main__":
    test_train()
    test_forecast()
    test_retrain()

