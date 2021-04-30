import pandas as pd
import csv
from datetime import datetime as dt

# Not my code, taken from the report ---------------------------------------

url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
df = pd.read_csv(url, sep='\,', na_values='.', engine='python')

url2 = "https://datagraphics.dckube.scilifelab.se/api/dataset/bbbaf64a25a1452287a8630503f07418.csv"
df_tests_Sweden = pd.read_csv(url2, sep='\,', na_values='.', engine='python')

countries = [
    'India', 'China', 'Iran', 'South Korea', 'South Africa', 'Kenya',
    'Bangladesh', 'Sweden', 'Norway', 'Germany', 'Italy', 'United Kingdom',
    'Brazil', 'United States', 'Canada', 'Australia'
]
df = df[df['location'].isin(countries)]
df = df.reset_index(drop=True)

df_countries = df.copy()

df = df.drop([
    'iso_code', 'continent', 'new_cases_per_million',
    'weekly_icu_admissions', 'weekly_icu_admissions_per_million',
    'weekly_hosp_admissions', 'weekly_hosp_admissions_per_million',
    'new_tests', 'total_tests_per_thousand', 'new_tests_per_thousand',
    'new_tests_smoothed', 'new_tests_smoothed_per_thousand', 'tests_units',
    'handwashing_facilities', 'hosp_patients', 'icu_patients',
], axis=1)

df.loc[df.location == 'Germany', 'extreme_poverty'] = 0.2

df_total_tests_Sweden = df_tests_Sweden.groupby(
    'date').sum().cumsum().reset_index()
df_Sweden = df.loc[df.location == 'Sweden']

df_Sweden = pd.merge(df_Sweden, df_total_tests_Sweden, how='left', on='date')
df_Sweden.drop('total_tests', axis=1, inplace=True)
df_Sweden = df_Sweden.rename(columns={'count': 'total_tests'})

df = df.loc[~(df.location == 'Sweden')]
df = df.append(df_Sweden, ignore_index=True)

df3 = pd.DataFrame(columns=df.columns)
for location in df.location.unique():
    # Gathering dataframe related to one location at a time
    df_location = df.loc[df['location'] == location].reset_index(drop=True)

    if df_location.location[0] == 'China':
        df_location = df_location[1:]
    # Setting total_tests = 1 the first time a case occures
    df_location.loc[(df_location.total_cases > 0).idxmax(), 'total_tests'] = 1

    # Interpolationg 'total tests'-column for each location:
    df_location['total_tests'].interpolate('linear', inplace=True)

    # Creating a new column for new tests based on interpolated total_tests-column
    df_location['new_tests'] = df_location.total_tests.diff()

    # Disregarding rows that have less than 100 confirmed cases:
    df_location = df_location.loc[df_location['total_cases'] > 100]

    # Disregarding last 9 rows of each location:
    df_location = df_location.iloc[:-10, :]

    # Sets Nan-values to zero for elements before first occurance in all columns defined by for-loop:
    for column_name in ['total_deaths', 'reproduction_rate']:
        index_of_first_valid = df_location[column_name].first_valid_index()
        if index_of_first_valid == None:
            continue
        df_location.loc[df_location.index < index_of_first_valid,
                        column_name] = 0

    # Takes the last valid value for 'reproduction_rate' and applies it to the last days
    df_location.loc[df_location.reproduction_rate.isna(),
                    'reproduction_rate'] = df_location.loc[
                        df_location.reproduction_rate.notna(),
                        'reproduction_rate'].iloc[-1]

    df_location = df_location.reset_index(drop=True)

    # Creates column based on index of dataframe
    df_location['days_after_100_cases'] = df_location.index
    # Appending the new dataframes df_location to df3
    df3 = df3.append(df_location, ignore_index=True)

df = df3

# Applying actual data to 'smoothed' columns where value is NaN
df.loc[df.new_cases_smoothed_per_million.isna(
), 'new_cases_smoothed_per_million'] = df.loc[:,
                                              'new_cases'] * 1000000 / df.loc[:,
                                                                              'population']
df.loc[df.new_deaths_smoothed_per_million.isna(
), 'new_deaths_smoothed_per_million'] = df.loc[:,
                                               'new_deaths'] * 1000000 / df.loc[:,
                                                                                'population']
df.loc[df.new_cases_smoothed.isna(),
       'new_cases_smoothed'] = df.loc[:, 'new_cases']
df.loc[df.new_deaths_smoothed.isna(),
       'new_deaths_smoothed'] = df.loc[:, 'new_deaths']

# Takes the last valid value for 'stringency_index' and applies it to the last days of the dataset for United Kingdom and Brazil.
last_stringency_index_United_Kingdom = df.loc[(
    (df.location == 'United Kingdom') & (df.stringency_index.notna())),
                                              'stringency_index'].iloc[-1]
df.loc[((df.location == 'United Kingdom') & (df.stringency_index.isna())),
       'stringency_index'] = last_stringency_index_United_Kingdom
last_stringency_index_Brazil = df.loc[((df.location == 'Brazil') &
                                       (df.stringency_index.notna())),
                                      'stringency_index'].iloc[-1]
df.loc[((df.location == 'Brazil') & (df.stringency_index.isna())),
       'stringency_index'] = last_stringency_index_Brazil

# Total deaths per million is NaN before the first death
df.loc[df['total_deaths_per_million'].isna(), 'total_deaths_per_million'] = 0

# My code -------------------------------------------------------------------------------------

# add two new columns (empty at first)
column_names = df.columns.values.tolist()
column_names.append("internal_flights")
column_names.append("international_arrivals")

df = df.reindex(columns=column_names)

# read data from flight dataset, update cells in new columns with relevant numbers
flights = {}
with open('flights.csv', 'r') as file:
    reader = csv.reader(file)
    for date, country, internal, arrivals in reader:
        df.loc[(df['location'] == country) & (df['date'] == date), ['internal_flights', 'international_arrivals']] = [internal, arrivals]

#---------------------------------------------------------------------------------------

# Converting date column to timestamp:
df.date = df.date.apply(lambda date: (dt.strptime(date, '%Y-%m-%d')))

# save cleaned and updated dataset as a new csv file
df.to_csv("updated_dataset.csv")