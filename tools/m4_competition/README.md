# M4 data processing

This Command Line Module formats the m4 competition dataset (found here: https://www.m4.unic.ac.cy/)
 into a format useful for forecasting experiments.
 
The m4 data set consists of a series of files, each containing timeseries data at different intervals:

```
* Daily-train.csv 
* Monthly-train.csv 
* Weekly-train.csv
* Hourly-train.csv 
* Quarterly-train.csv 
* Yearly-train.csv
```
each file contains a series of unlabelled timeseries sequences set with the same time interval.

It's worth keeping in mind that a few of these files are quite large (especially the `monthly` dataset), we strongly
encourage you to select a subset of the available variables for experimentation.

Like other formatting tools, the accompanying python script takes a CSV file, and formats it into the [standard timeseries format][stf]
 
 
  [stf]: https://github.com/algorithmiaio/OpenForecast/tree/master/tools/README.md#standardFormat