# Readme for Github users

<img src="https://algorithmia.com/algorithms/timeseries/OpenForecast/badge"></img>

Welcome to the OpenForecast project, github edition.

This algorithm supports multivariate, autoregressive forecasting with built-in model lifecycle management, neat huh!? 

Want to check out the Algorithmia docs? We've got an [alternate readme][algoreadme] to do just that, it also does a great job
at explaining what this algorithm can do - we recommend taking a quick look at it first if haven't heard of this before.

The remainder of this guide will walk you through the steps you'll need to get this project
up and running on your local system.

##  Requirements


This project has been tested on Ubuntu 16.04, and is not expected to work out-of-the-box in other environments.
This project uses python 3.5+, and it is not compatible with python 2.
This project requires an [Algorithmia][algo] account setup to interact with our hosted data resources.
The required pip packages are as follows:
```
algorithmia>=1.0.0,<2.0
six
numpy
https://download.pytorch.org/whl/nightly/cpu/torch_nightly-1.0.0.dev20181029-cp35-cp35m-linux_x86_64.whl
matplotlib
```
The pytorch 1.0.0 nightly wheel can be replaced with newer versions as required

**Note:** As this algorithm needs to interact with the Algorithmia [data API][data], you'll need to find your `ALGORITHMIA_API_KEY` so we can access the sample data.

We recommend setting the environment variable `ALGORITHMIA_API_KEY` to your api key using something like this:

```
export ALGORITHMIA_API_KEY=MY_API_KEY
source ~/.bashrc

```
And with that, we're ready to build a model.

## How to test

The algorithm should be fully runnable in an end2end fashion without any further changes, simply execute the following:

`python /src/OpenForecast_test.py`

and if you recieve no errors, the model development tests have succeeded.

If you wish, you can modify the file to use your own data API collection files, or alternatively local system files by prefixing the path with `local://`

Here's what the OpenForecast_test.py script looks like:

```python
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

```

This current execution system is subject to change, ideally we want a CLI that's as fully functional as the algorithmia API,
but for now this works.

Have any questions or comments? Feel free to create a git issue!



[algo]: https://algorithmia.com/
[test]: ../src/OpenForecast_test.py
[data]: https://docs.algorithmia.com/#data-api-specification
[dataspec]: https://algorithmia.com/data/hosted
[algoreadme]: README.md