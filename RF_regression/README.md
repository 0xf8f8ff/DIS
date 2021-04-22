Additional preprocessing specifically for RF algo:

1. RF does not work with null values, but there is no vaccination data in 2020, i.e., all vaccination related columns are filled with null values. We have to replace null values in vaccination columns with zeroes. (At the same time replacing any other data still missing after the initial cleaning).

2. For easier visualization and analysis we predict for one country at a time. The country name is provided as command line argument. Default country is Norway. We filter out rows that contain information for other countries.

3. We drop columns with index and location name, otherwise the algorithm would treat them as possible features slowing down the performance. 

4. We transform the date column into one-hot encoded seasons to use as a parameter in our RFR model.


