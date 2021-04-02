# Datasets

## Main datasets

1. [Daily COVID-19 data](https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv)
2. [Collection of monthly datasets of air traffic across the world](https://zenodo.org/record/4601479). We are only using datasets with records starting at 01.01.2020 as the original dataset has no data from before 2020.

## Helper datasets

1. [Data on testing for Sweden](https://datagraphics.dckube.scilifelab.se/api/dataset/bbbaf64a25a1452287a8630503f07418.csv)
2. [Two datasets with international codes for airports and countries from the same sourse](https://ourairports.com/data/)
- `countries.csv` contains ISO codes of the countries, `airports.csv` has all ICAO codes of the world airports

# Preprocessing

## Constructing lookup table with airport ICAO codes and country names (preprocess_ISO_codes.py)

After cleaning the original dataset we want to add two new columns with daily data: one for international flights inside the country, one for number of international flights arriving in the country on this day. The first problem is how to associate airport data from the second dataset with the country records in the main dataset. The main dataset has the name of the country and 3-letter ISO code in its records. The second dataset only has 4-letter ICAO code of the airport.

To associate airports with corresponding countries, we use helper datasets, one associating airport ICAO codes with 2-letter ISO codes of the countries, another one associating ISO codes with country names.

The code for this task is inside `preprocess_ISO_codes.py`. We only look at entries for relevant countries and for official airports. 

First mapper reads both helper datasets simultaniously. It yields ISO code as key and ICAO code or country name as value.

First reducer combines values together for each key. A combiner is unnecessary as datasets are small and it does not achieve any performance gains.

Second mapper reads all resulting lists and emits pairs of ICAO code and corresponding country name. Output has been recorded to a file: `lookupfile.csv`.

### Instructions:
- downloading two datasets:
1. ICAO code in column 2 (array element 1), ISO code of the country in column 9 (array element 8)
`wget https://ourairports.com/data/airports.csv`
2. ISO code in column 2 (elem 1), country name in column 3 (elem 2)
`wget https://ourairports.com/data/countries.csv`

- save datasets as codes0.csv, codes1.csv, copy to hdfs (folder name: data)

- running:
 ```
 python preprocess_ISO_codes.py -r hadoop hdfs:///data/codes* --output-dir hdfs:///data/lookup --no-output
 ```

Result will be saved to the /data/lookup folder

Check results: `hadoop fs -text /data/lookup/part* | less`

Merging the table:
```
touch lookupfile.csv
echo "icao, country" >> lookupfile.csv
cat part-00000 >> lookupfile.csv
cat part-00001 >> lookupfile.csv
```

Result: lookup table of format [ICAO airport code, country name]

## Using lookup table to process all dataset with flight data to a single dataset with only the relevant data (preprocess_flights.py)

Now we can use the constructed lookup table to preprocess all 4GB of flight data. 

Initializing mapper: loading lookup file and populating a dict.

Mapper: reads all datasets with flight data, emits lines with date, country of origin, country of destination. Resulting data is written into `allflights` folder.

### Instructions

When running the job pass lookup file as an argument.
```
python preprocess_flights.py --files lookupfile.csv -r hadoop hdfs:///data/flights/flightlist* --output-dir hdfs:///data/allflights --no-output
```

Check results: `hadoop fs -text /data/allflights/part* | less`

## Transforming the resulting dataset into a new dataset with records of structure [date, country, internal flights, international arrivals] (extract_flight_columns.py)

Mapper emits date and country as key, two element list indicating whether the flight was internal or from another country.

Reducer counts the number of internal flights and international arrivals for each key.
Output is recorded into `flights.csv`.

(TODO: we can try and add a combiner between the mapper and reducer to compare performance with and without the combiner)

### Instructions

`python extract_flight_columns.py -r hadoop hdfs:///data/flight_data_raw/part* --output-dir hdfs:///data/flights_processed --no-output`

## Cleaning the main dataset, adding data from the processed dataset as new columns (dataset ready, pandasjoin.py had dataframe-based code, mapreduce-based code is in progress)

Mapreduce code for this task is still being debugged. `pandasjoin.py` has code using pandas' dataframes. This codes gives same dataset as the mapreduce-based code. We can compare the performance of two code variants and use this info for performance tuning of the cluster.

All cleaning is applied as in the report unchanged, then two new columns added and populated with data from the flights dataset.

Resulting dataframe is saved into `updated_dataset.csv`.

All used and intermediate data (except the covid datasets) are on the cluster inside the `data` folder.

