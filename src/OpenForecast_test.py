#!/usr/bin/env python3
from src.OpenForecast import *
import os

def test_train():
    input = dict()
    input['mode'] = "train"
    input['data_path'] = "data://TimeSeries/GenerativeForecasting/m4_daily.json"
    input['model_output_path'] = "file://tmp/m4_daily_0.1.0.zip"
    input['training_time'] = 500
    input['model_complexity'] = 0.65
    input['forecast_length'] = 8
    result = apply(input)

    assert result['final_error'] <= 0.10
    assert os.path.isfile(result['model_output_path'])


def test_retrain():
    input = dict()
    input['mode'] = "train"
    input['data_path'] = "data://TimeSeries/GenerativeForecasting/m4_daily.json"
    input['model_input_path'] = "file://tmp/m4_daily_0.1.0.zip"
    input['model_output_path'] = "file://tmp/m4_daily_0.1.1.zip"
    result = apply(input)

    assert result['final_error'] <= 0.10
    assert os.path.isfile(result['model_output_path'])


def test_forecast():
    input = dict()
    input['mode'] = "forecast"
    input['model_input_path'] = "file://tmp/m4_daily_0.1.0.zip"
    input['graph_save_path'] = "file://tmp/my_forecast.png"
    input['forecast_length'] = 20
    input['data_path'] = "data://TimeSeries/GenerativeForecasting/m4_daily.json"
    input['io_noise'] = 0.05
    result = apply(input)

    assert os.path.isfile(result['graph_save_path'])

if __name__ == "__main__":
    test_train()
    test_forecast()
    test_retrain()
