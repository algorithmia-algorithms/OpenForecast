![sine wave splash](https://i.imgur.com/kDi9uIG.png)

GenerativeForecast is a **multivariate**, **incrementally trainable** auto-regressive forecasting algorithm.

## Introduction

What does all that mean? lets break it down.
* Multivatiate - This means that the algorithm can be trained to forecast multiple independent variables at once, this can be very useful for forecasting real world events like [earthquake forecasting][ef], along with more economically rewarding activities like [economic asset price prediction][econPred].
* Incrementally trainable - This algorithm can be incrementally trained with new data without needing to start from scratch. It's quite possible to automatically update your model or models on a daily/hourly/minute basis with new data that you can then use to quickly forecast the next N steps. It should also be mentioned that you don't _have_ to update your model before making a forecast! That's right, you can pass new data into the 'forecast' method and it will update the model state without running a backpropegation operation.
* Auto-regressive - Forecasting the future can be tricky, particularly when you aren't sure how far into the future you wish to look.  This algorithm uses it's previously predicted data points to help understand what the future looks like. For more information check out [this post][autoreg]

Powerful right? Lets get started in figuring out how this all works.

## Overview
This algorithm has two `modes`, **forecast** and **train**. To create a forecast you need a checkpoint model, which requires running a _train_ operation over at least part of your data.

### First time training
When training a model on your data for the first time, there are some important things to consider.
* First and foremost, the data _must_ be a file in csv format.
* In this csv file each column denotes an independent variable, and each row denotes a data point.
* Your data should be continuous, step wise operators make training more difficult.
* **Each point in your dataset must be in temporal order.**
Lets show you a quick example from a sine curve:

[initial training data for sine curve dataset][initsined]

Simple right? Lets also explore another dataset with two independent variables (this one is based on bitcoin price and transaction volume):

[initial training data for bitcoin dataset][initbitd]

Now we have an idea of what our data looks like, lets start exploring the other settings.


## Usage

### Input

_Describe the input fields for your algorithm. For example:_

| Parameter | Type | Description | Default or Required |
| --------- | ----------- | ----------- | ----------- |
| mode| String | tells the algorithm to anticipate a `train` or `forecast` operation. Please follow the guide below for more information. | Required |
| checkpoint_output_path | String | defines the output path for your trained model file. **must be a data connector URI(data://, s3://, dropbox://, etc)**| Required |
| data | String | the data connector URI(data://, s3://, dropbox://, etc) path pointing to training or evaluation data. Please follow the guide below for more information.| Not required, no default |

### Output

_Describe the output fields for your algorithm. For example:_

| Parameter | Description |
| --------- | ----------- |
| field     | Description of field |

## Examples

_Provide and explain examples of input and output for your algorithm._

[ef]: https://en.wikipedia.org/wiki/Earthquake_prediction
[econPred]: https://en.wikipedia.org/wiki/Stock_market_prediction
[autoreg]: https://dzone.com/articles/vector-autoregression-overview-and-proposals
[initsined]: https://gist.github.com/zeryx/00a84571fb3bfbfc4e08fdec2900b68f
[initbitd]: https://gist.github.com/zeryx/5d9a004ac10c4af702fc2a22dc3ad3f8