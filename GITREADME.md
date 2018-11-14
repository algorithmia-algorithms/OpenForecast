# Readme for Github users

<img src="https://algorithmia.com/algorithms/timeseries/OpenForecast/badge"></img>

So you've heard about this OpenForecast thing and you want to see if you can run it locally, you've come to the right place!


##  Requirements


This project has only been tested on Ubuntu 16.04, and is not expected to work out-of-the-box in other environments.
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

**Note:** As this algorithm needs to interact with the [Algorithmia][algo] data API, you'll need to find your `ALGORITHMIA_API_KEY` so we can access the sample data.


We recommend setting the environment variable `ALGORITHMIA_API_KEY` to your api key using something like this:

```
export ALGORITHMIA_API_KEY=MY_API_KEY
source ~/.bashrc

```
And with that, we're ready to build a model.

## How to train

For now, we recommend that you edit the [OpenForecast_test.py][test] file and edit the training function:

```python
def test_train():
    input = dict()
    input['mode'] = "train"
    input['data_path'] = "data://username/collection/dataset.json"
    input['model_output_path'] = "data://username/collection/model_0.1.0.zip"
    input['training_time'] = 300
    input['model_complexity'] = 0.65
    input['forecast_length'] = 10
    return apply(input)
```

At the very least, you should replace the following variables:
* `data_path`
* `model_output_path`

Once those are set for files in your algorithmia data collections, we can execute the training script.

`bash /src/OpenForecast_test.py`

## How to forecast

Again, lets take a look at the [OpenForecast_test.py][test] file, but this time, lets edit the forecast function:

```python
def test_forecast():
    input = dict()
    input['mode'] = "forecast"

    input['model_input_path'] = "data://username/collection/model_0.1.0.zip"
    input['graph_save_path'] = "data://username/collection/my_api_chart.png"
    input['data_path'] = "data://username/collection/dataset.json"
    input['forecast_length'] = 30
    input['io_noise'] = 0.05
    print(input)
    return apply(input)
```

As before, we have a few variables we need to replace:
* `data_path`
* `model_input_path`
* `graph_save_path`

Once those are changed, there's one last thing we need to do:

```python
"""Lets comment out test_train(), and uncomment test_forecast()"""
if __name__ == "__main__":
  result = test_forecast()
  # result = test_train()
  print(result)
```
Once again, we execute the script in the project root directory.

`bash /src/OpenForecast_test.py`

Have any questions or comments? Feel free to create a git issue!



[algo]: https://algorithmia.com/
[test]: src/OpenForecast_test.py