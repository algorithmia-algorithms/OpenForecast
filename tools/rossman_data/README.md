# Rossman data processing


This Command Line Module formats the Rossman sales data (found here: https://www.kaggle.com/c/cs3244-rossmann-store-sales/data)
 into a format useful for forecasting experiments.
 
The rossman data set has the first 66 days of data for over 1000 stores.
The python script takes the CSV file, and formats it into the [standard timeseries format][stf]
 
 
 
 
## Script usage

```
./python rossman_formatter.py --input_path="rossman_data.csv" --output_path="functional_rossman_data.json" --num_of_stores=4
```
input_path: the system path where your rossman training data (in csv format) is located.

output_path: the system path where you wish to place your formatted rossman training data

num_of_stores: the number of stores you wish to capture in your subsample.
 
 
 
 [stf]: https://github.com/algorithmiaio/OpenForecast/tree/master/tools/README.md#standardFormat