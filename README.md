OpenForecast is an open source **multivariate**, **portable** and **customizable** forecasting algorithm, you can see it in action [here][algolink].

<img src="https://i.imgur.com/BjGV1OX.png"></img>

## Introduction
What does all that mean? lets break it down.
* Multivatiate - This means that the algorithm can be trained to forecast *multiple independent variables at once*. This can be very useful for forecasting real world events like [earthquake forecasting][ef], along with more economically rewarding activities like [economic asset price prediction][econPred].
* Portable - We use [pytorch][pytorch] 1.0 here, this means we're able to package our recurrent model into a JIT compiled graph, which can be serialized and run almost anywhere!
* Auto-regressive - Forecasting the future can be tricky, particularly when you aren't sure how far into the future you wish to look.  This algorithm uses it's previously predicted data points to help understand what the future looks like. For more information check out [this post][autoreg].

Lets get started in figuring out how this all works.

## Getting Started Guide
This algorithm has two main `modes`, **forecast** and **train**. To create a forecast you need to create a checkpoint model, which you can make by running a _train_ operation over your input data.
The pipelines you create can be somewhat complex here so we're going to go over everything as much as we can.

### Training
First lets look at the **train** mode and how to get setup.
When training a model on your data, there are some important things to consider.
* First and foremost, the data should be processed into a compatible json format, check [here][rossman_example] for an example.
* Your data ideally is fully continuous, step wise operators make training more difficult. But as you can see in the above example, not necessary.
* **Each point in your dataset must be in temporal order.**

Some important parameters initial training parameters to consider:
* `model_complexity` - Describes your model's parameter density, higher values have more parameters. Ranges from 0 to 1+.
It's tough to overfit with our data augmentation strategies so using a larger number here for challenging datasets can certainly help, but it will take longer to train.
* `training_time` -   Defines how long in seconds we should spend training a model. The default is `450` as the default algorithm timeout is 500 seconds. Higher numbers can yield better results.
* `forecast_length` - In the training case, defines how far in the future the model should be able to predict for any given timestep. Larger lengths might be unstable and not train successfully, but smaller lengths might mean
the model isn't able to predict far enough in the future for your use case, defaults to `10`.

That is all we need to define our model.
Here are the remaining training settings that need to be defined, they mostly pertain to the training process itself and can be adjusted in future training steps:
* `io_noise` - We add gaussian noise to the input sequence during training, and can be kept during forecasting as well. We do this to prevent overfitting, and to force the model to learn large scale trends rather than micro-fluctuations. For most tasks `0.04` or `4%` noise is sufficient.
* `model_output_path` - the all important output path! We recommend a checkpoint name that contains version information and dataset information so you don't accidentally overwrite or misplace an important checkpoint in the future.

And after your model's trained, we output a forecast from the last timestep you provided!

##### Example IO
Input: 
```json
{  
   "model_output_path":"data://timeseries/generativeforecasting/rossman_5.zip",
   "io_noise":0.04,
   "data_path":"data://TimeSeries/GenerativeForecasting/rossman_5.json",
   "model_complexity": 0.65,
   "forecast_length": 10,
   "training_time": 30,
   "mode":"train"
   }
```

Output:

```json
{  
   "final_error":0.6678496301174164,
   "model_output_path":"data://timeseries/generativeforecasting/rossman_5.zip",
   "forecast":{  
      "sales for store #1":[  
         4131.84521484375,
         3950.352294921875,
         3744.051513671875,
         3555.768310546875,
         3409.609130859375,
         3318.1181640625,
         3287.380615234375,
         3305.02294921875,
         3359.5107421875,
         3445.07177734375
      ],
      "sales for store #2":[  
         5156.89306640625,
         4906.033203125,
         4684.193359375,
         4521.06005859375,
         4427.05126953125,
         4401.578125,
         4432.380859375,
         4507.53564453125,
         4622.65869140625,
         4766.45166015625
      ],
      "sales for store #3":[  
         7083.1044921875,
         6988.40283203125,
         6953.248046875,
         6966.27001953125,
         7018.67041015625,
         7098.595703125,
         7200.03662109375,
         7312.38720703125,
         7442.1142578125,
         7582.84130859375
      ],
      "sales for store #4":[  
         3845.955322265625,
         3630.4013671875,
         3418.158203125,
         3228.0283203125,
         3082.046630859375,
         2990.082275390625,
         2950.688232421875,
         2963.6240234375,
         3023.15771484375,
         3117.81494140625
      ],
      "sales for store #5":[  
         3826.99365234375,
         3701.860595703125,
         3587.1298828125,
         3486.552734375,
         3407.12158203125,
         3360.6328125,
         3355.103759765625,
         3387.739990234375,
         3451.078369140625,
         3536.541015625
      ]
   }
}
```

**Note**: If you provide a `model_input_path` during a training operation, the meta_data parameters from that model object
will be used to generate a new `Model` automatically, rather than forcing you to keep old model parameters.
This can be useful for when your algorithm experiences `concept drift` and needs to be retrained.

### Forecasting
So you've trained a model, gotten a basic forecast and now you want to start exploring your data in more depth with more forecasts.

The trained model has no knowledge of the data it's already seen, and doesn't store weights / residual vectors to keep things simple.
This is great, but it means you need to hold on to the data you used to train, or just append to it with new data as it comes in.

Here are some important forecasting variables to keep note of:
* forecast_size - Defines the number of steps into the future to forecast.
* io_noise - Defines how much noise is added to the initial memory state, and the attention vector. Larger values force the network to deviate faster but may reflect in a more accurate forcast.
* graph_save_path - If you'd like to have a pretty graph output like above, provide a data API URI here. Graphical output can be very useful for diagnosing and visualizing training issues.
* data_path - Just like in training, the path to your properly formatted json data.


##### Example IO
Input
```json
{  
   "model_input_path":"data://timeseries/generativeforecasting/rossman_5.zip",
   "mode":"forecast",
   "graph_save_path":"data://.algo/temp/forecast.png",
   "data_path":"data://TimeSeries/GenerativeForecasting/rossman_5_training.json"
}
```

Output
```json
{  
   "forecast":{  
      "sales for store #1":[  
         4101.65673828125,
         3914.804443359375,
         3699.373046875,
         3510.88134765625,
         3364.36474609375,
         3270.650634765625,
         3234.611083984375,
         3241.791015625,
         3284.655029296875,
         3359.50634765625
      ],
      "sales for store #2":[  
         5133.41455078125,
         4877.1953125,
         4657.724609375,
         4499.88720703125,
         4411.26953125,
         4380.919921875,
         4403.0673828125,
         4469.1572265625,
         4573.89892578125,
         4711.55859375
      ],
      "sales for store #3":[  
         7054.7734375,
         6959.3623046875,
         6936.23046875,
         6952.66357421875,
         6997.67626953125,
         7065.8779296875,
         7158.51220703125,
         7265.13525390625,
         7387.09619140625,
         7526.7685546875
      ],
      "sales for store #4":[  
         3830.94482421875,
         3611.1767578125,
         3397.71826171875,
         3212.484130859375,
         3070.39306640625,
         2975.947998046875,
         2930.5908203125,
         2934.201416015625,
         2981.682861328125,
         3063.127685546875
      ],
      "sales for store #5":[  
         3808.80859375,
         3681.504150390625,
         3570.080810546875,
         3476.306640625,
         3400.790771484375,
         3351.430908203125,
         3336.517333984375,
         3356.90380859375,
         3405.80029296875,
         3479.103271484375
      ]
   },
   "graph_save_path":"data://.algo/temp/forecast.png"
}
```

Did you notice the difference in results from the `train` step? That's because we didn't change the `io_noise` variable, 
if we set it to `0.0`, every subsequent forecast would be identical, given the same input.
## IO Schema

<a id="commonTable"></a>

### Common Table

| Parameter | Type | Description | Default if applicable |
| --------- | ----------- | ----------- | ----------- | ----------- |
| checkpoint_output_path | String | Defines the output path for your trained model file. | N/A |
| checkpoint_input_path | String | defines the input path for your existing model file. | N/A |
| data_path | String | The data connector URI(data://, s3://, dropbox://, etc) path pointing to training or evaluation data. | N/A |


<a id="forecastingTable"></a>

### Forecasting Table

#### Input

| Parameter | Type | Description | Default if applicable |
| --------- | ----------- | ----------- | ----------- |
| data_path | String | The path to your formatted data you wish to build a model on, must be stored on the `algorithmia data API`. | N/A |
| model_input_path | String | The data API path to the trained model you've previously built. |N/A|
| graph_save_path | String | The output path for your Monte Carlo forecast graph. | N/A |

#### Output

| Parameter | Type | Description |
| --------- | ----------- | ----------- |
| graph_save_apth | String | If you set a graph_save_path, then we successfully saved a graph at this data API location |
| forecast | Forecast | A forecast object containing information |

### Training Table

#### Input

| Parameter | Type | Description | Default if applicable |
| --------- | ----------- | ----------- | ----------- |
| training_time | Integer | Defines the number of seconds to continue training for, values above `450` require you to change the default algorithm timeout. | `500` |
| model_complexity | Float | A value between 0 and 1, defines how complex, or number of parameters to fit into the model. | `0.5` |
| data_path | String | The path to your formatted data you wish to build a model on, must be stored on the `data API`. | N/A |
| model_output_path | String | The output data API path where we plan to store the trained model, must be defined | N/A |
| model_input_path | String | If you wish to retrain your model using existing model parameters, provide the path to the existing model. | N/A |
| forecast_length| Int | The number of steps into the future we want our model to be able to predict, used in `train`ing and `forecast`ing. | `10` |
| io_noise | Float | Defines the percentage of Gaussian noise added to the training data to perturb the results, adding noise helps the model generalize to future trends. | `0.04` |

#### Output

| Parameter | Type |  Description |
| --------- | --------- | ----------- |
| model_output_path  | String | This is the path you provided as `model_output_path`, useful as a reminder |
| final_error | Float | The best generated model's error, the lower the better, ideally values below `0.01` is suggests a pretty good understanding of the sequence.
| forecast| Forecast | A forecast object containing information |

#### Example

``` json
{  
   "final_error":0.029636630788445473,
   "checkpoint_output_path":"data://timeseries/generativeforecasting/sinewave_model.zip",
   "forecast":{  
      "sales for store #1":[  
         4101.65673828125,
         3914.804443359375,
         3699.373046875,
         3510.88134765625,
         3364.36474609375,
         3270.650634765625,
         3234.611083984375,
         3241.791015625,
         3284.655029296875,
         3359.50634765625
      ],
      "sales for store #2":[  
         5133.41455078125,
         4877.1953125,
         4657.724609375,
         4499.88720703125,
         4411.26953125,
         4380.919921875,
         4403.0673828125,
         4469.1572265625,
         4573.89892578125,
         4711.55859375
      ],
      "sales for store #3":[  
         7054.7734375,
         6959.3623046875,
         6936.23046875,
         6952.66357421875,
         6997.67626953125,
         7065.8779296875,
         7158.51220703125,
         7265.13525390625,
         7387.09619140625,
         7526.7685546875
      ],
      "sales for store #4":[  
         3830.94482421875,
         3611.1767578125,
         3397.71826171875,
         3212.484130859375,
         3070.39306640625,
         2975.947998046875,
         2930.5908203125,
         2934.201416015625,
         2981.682861328125,
         3063.127685546875
      ],
      "sales for store #5":[  
         3808.80859375,
         3681.504150390625,
         3570.080810546875,
         3476.306640625,
         3400.790771484375,
         3351.430908203125,
         3336.517333984375,
         3356.90380859375,
         3405.80029296875,
         3479.103271484375
      ]
   }
}
```


[ef]: https://en.wikipedia.org/wiki/Earthquake_prediction
[econPred]: https://en.wikipedia.org/wiki/Stock_market_prediction
[autoreg]: https://dzone.com/articles/vector-autoregression-overview-and-proposals
[algolink]: https://algorithmia.com/algorithms/TimeSeries/OpenForecast
[rossman_example]: https://github.com/algorithmiaio/OpenForecast/tree/master/tools#the-standard-timeseries-format
[pytorch]: https://pytorch.org/get-started/locally/
[gitreadme]: GITREADME.d