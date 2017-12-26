![sine wave splash](https://i.imgur.com/kDi9uIG.png)

GenerativeForecast is a **multivariate**, **incrementally trainable** auto-regressive forecasting algorithm.

## Introduction
What does all that mean? lets break it down.
* Multivatiate - This means that the algorithm can be trained to forecast multiple independent variables at once, this can be very useful for forecasting real world events like [earthquake forecasting][ef], along with more economically rewarding activities like [economic asset price prediction][econPred].
* Incrementally trainable - This algorithm can be incrementally trained with new data without needing to start from scratch. It's quite possible to automatically update your model or models on a daily/hourly/minute basis with new data that you can then use to quickly forecast the next N steps. It should also be mentioned that you don't _have_ to update your model before making a forecast! That's right, you can pass new data into the 'forecast' method and it will update the model state without running a backpropegation operation.
* Auto-regressive - Forecasting the future can be tricky, particularly when you aren't sure how far into the future you wish to look.  This algorithm uses it's previously predicted data points to help understand what the future looks like. For more information check out [this post][autoreg].

Powerful right? Lets get started in figuring out how this all works.

## Getting Started Guide
This algorithm has two main `modes`, **forecast** and **train**. To create a forecast you need to create a checkpoint model, which you can make by running a _train_ operation over your input data.
The pipelines you create can be somewhat complex here so we're going to go over everything as much as we can.

### Training
First lets look at the **train** mode and how to get setup.
<a id="initialTraining"></a>
#### First time training
When training a model on your data for the first time, there are some important things to consider.
* First and foremost, the data _must_ be a file in csv format.
* In this csv file each column denotes an independent variable, and each row denotes a data point.
* Your data should be continuous, step wise operators make training more difficult.
* If you'd like your variables to be described in forecasts, be sure to start your training data csv with headers that define your input.
* **Each point in your dataset must be in temporal order.**

Let's take a quick look with an example of a sine curve:

[initial training data for sine curve dataset][initsined]

Simple right? Lets also explore another dataset with two independent variables (this one is based on bitcoin price and transaction volume):

[initial training data for bitcoin dataset][initbitd]

Notice the headers? **You only have to define headers when training a brand new model, the network file itself will store your headers to keep things simple.** Don't worry if all your csv data has headers, our algorithm is smart enough to figure that out! What if you don't have headers? No problem, the algorithm has default variable names to use if they're missing.


Now that we have our data all ready, lets take a look at our [training parameters table](#trainingTable).

Some important parameters initial training parameters to consider:
* `layer_width` - Defines how much knowledge your model is able to grasp. It's tough to overfit with our data augmentation strategies so using a larger number here for challenging datasets can certainly help.
* `attention_width` -  Large attention vectors can help improve your models recall of events, but potentially at the expense of learning how to manage that internally in LSTM layers. *In the initial release this is a hard attention vector pointed to the last N steps, where N is the width of the vector, this may change in the future.*
* `future_beam_width` - This provides us a tool to force the model to predict future events besides just the first step. We use a custom loss function to take large beams into consideration.

That is all we need to define our model.
Here are the remaining training settings that need to be defined, they mostly pretain to the training process itself and can be adjusted in future training steps:
* `input_dropout` - How can we make sure that our training process is kept on task? We do this by forcing the model to stoichastically feed it it's own predictions as input! This keeps the training model focused on auto-regressive forcasting, and along with `io_noise` prevents overfitting.
* `io_noise` - We add gaussian noise to the input and output for our network during training, and can be kept during forecasting as well. We do this to prevent overfitting, and to force the model to learn large scale trends rather than micro-fluctuations. For most tasks `0.04` or `4%` noise is sufficient.
* `checkpoint_output_path` - the all important output path! We recommend a checkpoint name that contains version information and dataset information so you don't accidentally overwrite or misplace an important checkpoint in the future.

##### Example IO
Input: 
``` json
{  
 "checkpoint_output_path":"data://timeseries/generativeforecasting/sinewave_v1.0_t0.t7",
   "layer_width":55,
   "iterations":2,
   "io_noise":0.04,
   "data_path":"data://TimeSeries/GenerativeForecasting/sinewave_v1.0_t0.csv",
   "attention_beam_width":55,
   "input_dropout":0.4,
   "mode":"train",
   "future_beam_width":25
}
```

Output:

``` json
{  
   "final_error":0.03891853988170624,
   "model_save_path":"data://timeseries/generativeforecasting/sinewave_v1.0_t0.t7"
}
```

For our initial training we specify all network definition parameters, along with a checkpoint output path, and a data path.
Keep note of that saved filepath, we're going to need that later.

<a id="incrTraining"></a>
#### Incremental Training

So you have a model that you've already trained already, and it's been giving you great forecasts. But you've noticed new trends evolving in your timeseries that your model isn't able to predict. Wouldn't it be great if there was a way to incrementally update your model? There is! :smile: 

When you already have a trained model, you can incrementally retrain it by simply providing that model URI with the `checkpoint_input_path` key in your input, that's it! All network definition parameters are preserved so there's no need to write them all out again. 
Here is a simple list of parameters you can adjust during incremental training:

* `input_dropout` - How can we make sure that our training process is kept on task? We do this by forcing the model to stoichastically feed it it's own predictions as input. This keeps the training model focused on auto-regressive forcasting, and along with `io_noise`, prevents overfitting.
* `io_noise` - We add gaussian noise to the input and output for our network during training, and can be kept during forecasting as well. We do this to prevent overfitting, and to force the model to learn large scale trends rather than micro-fluctuations. For most tasks `0.04` or `4%` noise is sufficient.

For more information on the schema, please take a look at the [training IO table](#trainingTable)

#### Example IO

Input:
``` json
{ 
  "checkpoint_input_path":"data://timeseries/generativeforecasting/sinewave_v1.0_t0.t7",
  "checkpoint_output_path":"data://timeseries/generativeforecasting/sinewave_v1.0_t1.t7",
   "iterations":2,
   "io_noise":0.04,
   "data_path":"data://TimeSeries/GenerativeForecasting/sinewave_v1.0_t1.csv",
   "input_dropout":0.4,
   "mode":"train"
}
```

Output:
``` json
{  
   "final_error":0.02383182378470221,
   "checkpoint_output_path":"data://timeseries/generativeforecasting/sinewave_v1.0_t1.t7"
}
```

And just like that you've updated your model to detect new trends.

<a id="forecasting"></a>
### Forecasting
So you've trained a model and now you want to start exploring your data, lets take a look at how to make forecasts.

There are two ways to create a forecast, by using an up-to-date model, or by incrementally updating an existing model (no gradient updates) with new data. We call these two methods `tip-of-checkpoint forecasting` and `incremental update forecasting`. What's the difference? Simply by providing a `data_path` URL you're automatically telling the algorithm you'd like to create an `incremental inference forecast`! Let's take a look at some key parameters for forecasting:

* forecast_size - Defines the number of steps into the future to forecast.
* iterations - Defines the number of independently calculated forecast operations to perform, each forecast is initialized by perturbing the memory state of the checkpoint with `io_noise` to generate a monte carlo forecast envelope.
* io_noise - Defines how much noise is added to the initial memory state, and the attention vector. Larger values force the network to deviate faster but may reflect in a more accurate forcast.
* graph_save_path - If you'd like to have a pretty graph output like above, provide a data API URI here. Graphical output can be very useful for diagnosing and visualizing training issues.

For more information, take a look at the [Forecasting IO table](#forecastingTable)


Lets take a quick look at a `tip-of-checkpoint` example:

#### tip-of-checkpoint IO
![my_sinegraph_t0](https://i.imgur.com/hhB6fCb.png)

Input:

``` json
{
    "mode":"forecast",
    "checkpoint_input_path":"data://timeseries/generativeforecasting/sinewave_v1.0_t0.t7",
    "graph_save_path":"data://.algo/timeseries/generativeforecasting/temp/my_sinegraph_t0.png",
    "forecast_size": 10,
    "iterations": 25,
    "io_noise": 0.05
}
```

Output:

``` json
{  
   "saved_graph_path":"data://.algo/timeseries/generativeforecasting/temp/my_sinegraph_t0.png",
   "envelope":[  
      {  
         "mean":[  
            -0.2524857783317566,
            -0.2559856462478638,
            -0.25597516059875486,
            -0.2572448682785034,
            -0.2587031388282776,
            -0.25668447971343994,
            -0.2572889566421509,
            -0.2558885383605957,
            -0.2552607440948486,
            -0.2559203052520752
         ],
         "second_deviation":{  
            "lower_bound":[  
               -0.2795769316136562,
               -0.27075492995672173,
               -0.26555003747913497,
               -0.2656836182323437,
               -0.268803374512105,
               -0.26423675519300543,
               -0.2667612249165839,
               -0.263370684851367,
               -0.2649338482841223,
               -0.26581953863931707
            ],
            "upper_bound":[  
               -0.22539462504985702,
               -0.24121636253900586,
               -0.24640028371837475,
               -0.24880611832466307,
               -0.24860290314445022,
               -0.24913220423387447,
               -0.24781668836771792,
               -0.2484063918698244,
               -0.2455876399055749,
               -0.24602107186483335
            ]
         },
         "variable":"y(t)",
         "standard_deviation":[  
            0.013545576640949792,
            0.007384641854428972,
            0.004787438440190054,
            0.004219374976920166,
            0.005050117841913699,
            0.0037761377397827396,
            0.004736134137216495,
            0.003741073245385651,
            0.00483655209463686,
            0.004949616693620926
         ],
         "first_deviation":{  
            "lower_bound":[  
               -0.2660313549727064,
               -0.2633702881022928,
               -0.2607625990389449,
               -0.26146424325542356,
               -0.26375325667019134,
               -0.2604606174532227,
               -0.2620250907793674,
               -0.25962961160598136,
               -0.26009729618948546,
               -0.26086992194569614
            ],
            "upper_bound":[  
               -0.23894020169080682,
               -0.24860100439343483,
               -0.25118772215856483,
               -0.25302549330158325,
               -0.2536530209863639,
               -0.2529083419736572,
               -0.2525528225049344,
               -0.2521474651152101,
               -0.25042419200021176,
               -0.2509706885584543
            ]
         }
      }
   ]
}
```

So in this example we have the [envelope](#envelopeTable) coordinates defined as multiple lists of `forecast_size` in length.

Now lets take a look at an example with `incremental update forecasting`:

#### incremental update IO

![my_sinegraph_t1_trained](https://i.imgur.com/CIpIEZW.png)

Input: 
``` json
{
    "mode":"forecast",
    "checkpoint_input_path":"data://timeseries/generativeforecasting/sinewave_v1.0_t0.t7",
    "checkpoint_output_path":"data://timeseries/generativeforecasting/sinewave_v1.0_t1_inf.t7
    "data_path":"data://timeseries/generativeforecasting/sinewave_v1.0_t1.csv"
    "graph_save_path":"data://.algo/timeseries/generativeforecasting/temp/my_sinegraph.png",
    "forecast_size": 10,
    "iterations": 25,
    "io_noise": 0.05
}
```
Output":
``` json
{  
   "saved_graph_path":"data://timeseries/generativeforecasting/my_sinegraph_t1.png",
   "checkpoint_output_path":"data://timeseries/generativeforecasting/sinewave_v1.0_t1_inf.t7",
   "envelope":[  
      {  
         "mean":[  
            0.008691980838775634,
            0.03607967376708984,
            0.05621590614318848,
            0.06999496936798096,
            0.08218138694763183,
            0.08151638984680176,
            0.07954122066497803,
            0.07741095066070557,
            0.06946793556213379,
            0.056818246841430664
         ],
         "first_deviation":{  
            "lower_bound":[  
               -0.011594484467470512,
               0.018994830103149025,
               0.04193864726039028,
               0.05866774251829453,
               0.07278376819517646,
               0.07200866910333781,
               0.06691972572770245,
               0.06318318488777479,
               0.05722540719017249,
               0.04463311867402814
            ],
            "upper_bound":[  
               0.02897844614502178,
               0.053164517431030664,
               0.07049316502598668,
               0.08132219621766738,
               0.0915790057000872,
               0.09102411059026572,
               0.0921627156022536,
               0.09163871643363634,
               0.08171046393409509,
               0.06900337500883319
            ]
         },
         "variable":"Y(t)",
         "second_deviation":{  
            "lower_bound":[  
               -0.03188094977371666,
               0.001909986439208207,
               0.027661388377592085,
               0.047340515668608106,
               0.06338614944272108,
               0.06250094835987385,
               0.054298230790426893,
               0.048955419114843995,
               0.04498287881821118,
               0.03244799050662561
            ],
            "upper_bound":[  
               0.04926491145126793,
               0.07024936109497149,
               0.08477042390878486,
               0.0926494230673538,
               0.10097662445254257,
               0.10053183133372967,
               0.10478421053952916,
               0.10586648220656714,
               0.0939529923060564,
               0.08118850317623572
            ]
         },
         "standard_deviation":[  
            0.020286465306246147,
            0.017084843663940818,
            0.014277258882798197,
            0.011327226849686425,
            0.009397618752455369,
            0.009507720743463956,
            0.012621494937275568,
            0.014227765772930783,
            0.012242528371961305,
            0.012185128167402526
         ]
      }
   ]
}
```

The graphs are different! This is beacuse when you pass a `data_path` as input, it automatically updates the model state to the end of that `data_path`. 
*When incrementally updating, always ensure that your next dataset is in sequential order from the previous dataset.*

What happens if we reuse that saved model and run a `tip-of-checkpoint` forecast? Well let's find out!

![inferred graph](https://i.imgur.com/pH30mNl.png)


``` json
{  
   "iterations":25,
   "mode":"forecast",
   "graph_save_path":"data://timeseries/generativeforecasting/my_sinegraph_t1_inferred.png",
   "forecast_size":100,
   "checkpoint_input_path":"data://timeseries/generativeforecasting/sinewave_v1.0_t1_inf.t7",
   "io_noise":0.05
}
```

It works! No backpropegation required for simple updates like this. If your datas signals or trends do change or drift overtime, it is highly recommended to run a [incremental training](#incrTraining) operation perioidically to ensure forecast accuracy.

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
[initsined]: https://gist.github.com/zeryx/00a84571fb3bfbfc4e08fdec2900b68f
[initbitd]: https://gist.github.com/zeryx/5d9a004ac10c4af702fc2a22dc3ad3f8