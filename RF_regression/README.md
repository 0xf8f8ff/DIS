Additional preprocessing specifically for RF algo:

1. RF does not work with null values, but there is no vaccination data in 2020, i.e., all vaccination related columns are filled with null values. We have to replace null values in vaccination columns with zeroes. (At the same time replacing any other data still missing after the initial cleaning).

2. For easier visualization and analysis we predict for one country at a time. The country name is provided as command line argument. Default country is Norway. We filter out rows that contain information for other countries.

3. We drop columns with index and location name, otherwise the algorithm would treat them as possible features slowing down the performance. 

4. We transform the date column into month number.

5. Additionally, we split the dataset into the training and testing subsets. All data for 2020 goes into training set, all data for 2021 - into testing.


```
python rf_mr.py --files rf_testing.csv,rf_training.csv --hadoop-streaming-jar /usr/hdp/current/hadoop-mapreduce-client/hadoop-streaming.jar -r hadoop hdfs:///testdata/dummy --output-dir hdfs:///rfregression --no-output
```