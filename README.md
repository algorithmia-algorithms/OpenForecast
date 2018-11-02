OpenForecast is an open source **multivariate**, **portable** and **customizable** forecasting algorithm, you can see it in action [here][algolink].

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
* `model_complexity` - Defines how much knowledge your model is able to grasp. 
It's tough to overfit with our data augmentation strategies so using a larger number here for challenging datasets can certainly help, but it will take longer to train.
* `training_time` -   Defines how long in seconds we should spend training a model. The default is `450` as the default algorithm timeout is 500 seconds. Higher numbers can yield better results.
* `forecast_length` - In the training case, defines how far in the future the model should be able to predict for any given timestep. Larger lengths might be unstable and not train successfully, but smaller lengths might mean
the model isn't able to predict far enough in the future for your use case, defaults to `10`.

That is all we need to define our model.
Here are the remaining training settings that need to be defined, they mostly pertain to the training process itself and can be adjusted in future training steps:
* `io_noise` - We add gaussian noise to the input sequence during training, and can be kept during forecasting as well. We do this to prevent overfitting, and to force the model to learn large scale trends rather than micro-fluctuations. For most tasks `0.04` or `4%` noise is sufficient.
* `checkpoint_output_path` - the all important output path! We recommend a checkpoint name that contains version information and dataset information so you don't accidentally overwrite or misplace an important checkpoint in the future.

And after your model's trained, we output a forecast from the last timestep you provided!

##### Example IO
Input: 
``` json
{  
 "model_input_path":"data://timeseries/generativeforecasting/rossman_5.zip",
   "io_noise":0.04,
   "data_path":"data://TimeSeries/GenerativeForecasting/rossman_5.json",
   "model_complexity": 0.65,
   "forecast_length": 10,
   "training_time": 500
   "mode":"train",
   "future_beam_width":25
}
```

Output:

``` json
{  
   "final_error":0.03891853988170624,
   "model_save_path":"data://timeseries/generativeforecasting/rossman_5.zip"
}
```

For our initial training we specify all network definition parameters, along with a checkpoint output path, and a data path.
Keep note of that saved filepath, we're going to need that later.

### Forecasting
So you've trained a model and now you want to start exploring your data, lets take a look at how to make forecasts.

The trained model has no knowledge of the data it's already seen, and doesn't store weights / residual vectors to keep things simple.
This is great, but it means you need to hold on to the data you used to train, or just append to it with new data as it comes in.

Here are some important forecasting variables to keep note of:
* forecast_size - Defines the number of steps into the future to forecast.
* io_noise - Defines how much noise is added to the initial memory state, and the attention vector. Larger values force the network to deviate faster but may reflect in a more accurate forcast.
* graph_save_path - If you'd like to have a pretty graph output like above, provide a data API URI here. Graphical output can be very useful for diagnosing and visualizing training issues.
* data_path - Just like in training, the path to your properly formatted json data.


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
| iterations | Integer | The number of independent forecast operations used to create your monte carlo envelope | `10` |
| graph_save_path | String | The output path for your Monte Carlo forecast graph. | N/A |

#### Output

| Parameter | Type | Description |
| --------- | ----------- | ----------- |
| checkpoint_output_path | String | This is the path you provided as `checkpoint_output_path`, useful as a reminder |
| envelope | List[Envelope] | A list of Envelope objects for each dimension of your data, see below for more info. |

<a id="envelopeTable"></a>
#### Envelope Object
Each independent variable has it's own Envelope object, with the variable name defined by `variable`.

| Parameter | Type | Description | 
| --------- | --------- | --------- |
| variable | String | The name of the variable for this dimension, defined during initial training from your csv header. |
| mean | List[Float] | The mean for each point in your forecast, for this variable |
| standard_deviation | List[Float] | The Standard deviation for each point in your forecast, for this variable. |
| first_deviation | Deviation | The upper and lower bounds for the first standard deviation from the mean, for this variable. |
| second_deviation | Deviation | The upper and lower bounds for the second standard deviation from the mean, for this variable. |


#### Deviation Object
| Parameter | Type | Description | 
| --------- | --------- | --------- |
| upper_bound | List[Float] | The upper bound values for this deviation. |
| lower_bound | List[Float] | The lower bound values for this deviation. |
 
 
##### Example

``` json
{  
   "envelope":[  
      {  
         "second_deviation":{  
            "upper_bound":[  
               -0.9742458981517674,
               -0.9500016416591704,
               -0.9139700686053683,
               -0.8639122314814361,
               -0.8069765949845458,
               -0.7560190278939466,
               -0.6934241707923614,
               -0.6294399237192251,
               -0.5676434989916705,
               -0.5101172068230402
            ],
            "lower_bound":[  
               -1.0399963026996728,
               -0.9996776853519379,
               -0.951504861204202,
               -0.9103988246556488,
               -0.8559860467314577,
               -0.7970966499396105,
               -0.7433021895286835,
               -0.685673487230482,
               -0.6241579154980756,
               -0.5546375129111518
            ]
         },
         "mean":[  
            -1.0071211004257201,
            -0.9748396635055542,
            -0.9327374649047852,
            -0.8871555280685425,
            -0.8314813208580017,
            -0.7765578389167785,
            -0.7183631801605225,
            -0.6575567054748536,
            -0.5959007072448731,
            -0.532377359867096
         ],
         "standard_deviation":[  
            0.016437601136976357,
            0.012419010923191857,
            0.00938369814970841,
            0.011621648293553192,
            0.012252362936727983,
            0.010269405511416,
            0.012469504684080545,
            0.014058390877814235,
            0.014128604126601265,
            0.011130076522027917
         ],
         "variable":"0",
         "first_deviation":{  
            "upper_bound":[  
               -0.9906834992887438,
               -0.9624206525823623,
               -0.9233537667550767,
               -0.8755338797749893,
               -0.8192289579212738,
               -0.7662884334053626,
               -0.7058936754764419,
               -0.6434983145970393,
               -0.5817721031182719,
               -0.5212472833450681
            ],
            "lower_bound":[  
               -1.0235587015626966,
               -0.987258674428746,
               -0.9421211630544936,
               -0.8987771763620956,
               -0.8437336837947297,
               -0.7868272444281945,
               -0.7308326848446031,
               -0.6716150963526678,
               -0.6100293113714743,
               -0.5435074363891239
            ]
         }
      }
   ],
   "saved_graph_path":"data://timeseries/generativeforecasting/sinewave_forecast.png"
}

```

<a id="trainingTable"></a>
         
### Training Table

#### Input

| Parameter | Type | Description | Default if applicable |
| --------- | ----------- | ----------- | ----------- |
| iterations | Integer | Defines the number of iterations per epoch for training your model. Bigger numbers makes training take longer, but can yield better results. | `10` |
| layer_width | Integer | Defines your models layer width, layer depth is automatically determined by the number of independent variables in your dataset. | `51` |
| attention_width | Integer | Defines your networks hard attention beam width. Larger beams are useful for complex data models.| `25` |
| future_beam_width | Integer | Similar to the `attention_width` but this defines how many steps we predict in one training step.| `10` |
| input_dropout | Float | This defines the percentage of input that we "drop out" during training. | `0.45` |
| io_noise | Float | Defines the percentage of Gaussian noise added to the training data to perturb the results. Both noise and input_dropout help the model generalize to future trends. | `0.04` |

#### Output

| Parameter | Type |  Description |
| --------- | --------- | ----------- |
| checkpoint_output_path  | String | This is the path you provided as `checkpoint_output_path`, useful as a reminder |
| final error | Float | The best generated model's error, the lower the better.

#### Example

``` json
{  
   "final_error":0.029636630788445473,
   "checkpoint_output_path":"data://timeseries/generativeforecasting/sinewave_model.t7"
}
```

### Frequently Asked Questions

#### Why are my forecast images always scaled between 0 and 1?
Great question! We do this so that multivariate graphs are on the same scale. If you have 2 or more independent variables it can be quite difficult to represent them in their original domain. Rest assured that the `envelope` returns denormalized data.

#### How do I know what parameters to use for attention_width, future_beam_width, etc?
Unfortunately there is no `one size fits all` solution here, it's highly dependent on your data! The default network parameter values work pretty well for us, but we recommend exploring your data by creating multiple initial models with different network parameters and seeing what works best for you.


#### I know how well my model performs during training, but how can I calculate my model's generative forecast accuracy?
With input_dropout we can get pretty close to a real accuracy measurement, but for a more explicit calculation be on the lookout for a sister algorithm named `ForecastAccuracy`.

[ef]: https://en.wikipedia.org/wiki/Earthquake_prediction
[econPred]: https://en.wikipedia.org/wiki/Stock_market_prediction
[autoreg]: https://dzone.com/articles/vector-autoregression-overview-and-proposals
[algolink]: https://algorithmia.com/algorithms/TimeSeries/OpenForecast
[rossman_example]: https://github.com/algorithmiaio/OpenForecast/tree/master/tools/rossman
[pytorch]: https://pytorch.org/get-started/locally/