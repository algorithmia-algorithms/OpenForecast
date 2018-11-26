# Tools

This directory contains a series of data preparation packages designed to take raw data from a third party resource, and convert it into a
format that the OpenForecast algorithm can handle.

Currently we have examples for:
* [The Rossman store dataset][rossman]


# The Standard Timeseries Format

For our timeseries algorithms, data must be formatted into the following `json object`.

```
{
  "tensor": List[List[FloatVariable]]
  "key_variables": List[Column]
}
```
tensor: The 1d+ timeseries data, compatible with numpy's ndarray, formatted with
the .tolist() function for serialization. Values must all be floating point.

key_variables: Important, or key variables that you wish to forecast. If there are other 
variables present, they are thus considered `Auxilliary`, and are not forecasted (think open/close states, weather data, etc).

```
Variable: {"index": Int, "header":str}
```

index: the column number for this forecastable column

header: the name or title used to describe this column,
used to decorate the forecast and graph results.

### Example of a STF data file
```json
{   "key_variables":[
      {  
         "index":0, "header":"sales for store #1"
      }
   ],
   "tensor":[
      [5919, 624, 1, 5, 0, 0, 1],
      [4775, 539, 1, 4, 0, 0, 1],
      [6032, 720, 1, 3, 0, 0, 1],
      [5258, 575, 1, 2, 0, 0, 1],
      [5931, 638, 1, 1, 0, 0, 1],
      [0, 0, 0, 7, 0, 0, 0],
      [2575, 326, 1, 6, 0, 0, 0],
      [3206, 418, 1, 5, 0, 0, 0],
      [3728, 484, 1, 4, 0, 0, 0],
      [4524, 606, 1, 3, 0, 0, 0],
      [3955, 491, 1, 2, 0, 0, 0],
      [4082, 528, 1, 1, 0, 0, 0],
      [0, 0, 0, 7, 0, 0, 0],
      [2883, 340, 1, 6, 0, 0, 0],
      [5456, 573, 1, 5, 0, 0, 1],
      [5810, 632, 1, 4, 0, 0, 1],
      [6366, 700, 1, 3, 0, 0, 1],
      [6444, 695, 1, 2, 0, 0, 1],
      [7237, 689, 1, 1, 0, 0, 1],
      [0, 0, 0, 7, 0, 0, 0],
      [3024, 346, 1, 6, 0, 0, 0],
      [4762, 524, 1, 5, 0, 0, 1],
      [5246, 567, 1, 4, 0, 0, 1],
      [5945, 670, 1, 3, 0, 0, 1],
      [6696, 727, 1, 2, 0, 0, 1],
      [7308, 755, 1, 1, 0, 0, 1],
      [0, 0, 0, 7, 0, 0, 0],
      [3132, 350, 1, 6, 0, 0, 0],
      [4137, 543, 1, 5, 0, 1, 0],
      [0, 0, 0, 4, 0, 1, 0],
      [2269, 252, 1, 3, 0, 1, 0],
      [5513, 632, 1, 2, 0, 1, 0],
      [5999, 662, 1, 1, 0, 1, 0],
      [0, 0, 0, 7, 0, 0, 0],
      [3934, 457, 1, 6, 0, 0, 0],
      [0, 0, 0, 5, 3, 1, 0],
      [0, 0, 0, 4, 3, 1, 0],
      [2437, 274, 1, 3, 0, 1, 0],
      [7557, 816, 1, 2, 0, 1, 0],
      [9027, 924, 1, 1, 0, 1, 0],
      [0, 0, 0, 7, 0, 0, 0],
      [4553, 495, 1, 6, 0, 0, 0],
      [7574, 718, 1, 5, 0, 0, 1],
      [8117, 777, 1, 4, 0, 0, 1],
      [9365, 909, 1, 3, 0, 0, 1],
      [9499, 866, 1, 2, 0, 0, 1],
      [10419, 909, 1, 1, 0, 0, 1],
      [0, 0, 0, 7, 0, 0, 0],
      [3375, 350, 1, 6, 0, 0, 0],
      [4407, 497, 1, 5, 0, 0, 0],
      [4629, 539, 1, 4, 0, 0, 0],
      [5830, 717, 1, 3, 0, 0, 0],
      [5264, 596, 1, 2, 0, 0, 0],
      [4821, 616, 1, 1, 0, 0, 0],
      [0, 0, 0, 7, 0, 0, 0],
      [2914, 386, 1, 6, 0, 0, 0],
      [6385, 676, 1, 5, 0, 0, 1],
      [6661, 670, 1, 4, 0, 0, 1],
      [7533, 769, 1, 3, 0, 0, 1],
      [7664, 751, 1, 2, 0, 0, 1],
      [8674, 783, 1, 1, 0, 0, 1],
      [0, 0, 0, 7, 0, 0, 0],
      [3424, 372, 1, 6, 0, 0, 0],
      [6062, 608, 1, 5, 0, 0, 1],
      [6119, 644, 1, 4, 0, 0, 1],
      [7588, 828, 1, 3, 0, 0, 1]
   ]
}
```


[rossman]: https://www.kaggle.com/c/cs3244-rossmann-store-sales/data
