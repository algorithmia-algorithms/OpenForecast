![sine wave splash](https://i.imgur.com/kDi9uIG.png)

GenerativeForecast is a **multivariate**, **incrementally trainable** auto-regressive forecasting algorithm.

## Introduction

What does all that mean? lets break it down.
* Multivatiate - This means that the algorithm can be trained to forecast multiple independent variables at once, this can be very useful for forecasting real world events like [earthquake forecasting][ef], along with more economically rewarding activities like [economic asset price prediction][econPred].
* Incrementally trainable - This algorithm can be incrementally trained with new data without needing to start from scratch. It's quite possible to automatically update your model or models on a daily/hourly/minute basis with new data that you can then use to quickly forecast the next N steps. It should also be mentioned that you don't _have_ to update your model before making a forecast! That's right, you can pass new data into the 'forecast' method and it will update the model state without running a backpropegation operation.
* Auto-regressive - Forecasting the future can be tricky, particularly when you aren't sure how far into the future you wish to look.  This algorithm uses it's previously predicted data points to help understand what the future looks like. For more information check out [this post][autoreg]



## Usage

### Input

_Describe the input fields for your algorithm. For example:_

| Parameter | Description |
| --------- | ----------- |
| field     | Description of field |

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
