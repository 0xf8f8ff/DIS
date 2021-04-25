import pandas as pd
import sys
from datetime import datetime as dt


location = "all"
if len(sys.argv) == 2:
    location = str(sys.argv[1])


data = pd.read_csv('updated_dataset.csv')

if location != "all":
    print("Before condition records: " + str(data.shape[0]))
    data = data[data['location'] == location]
    print("After condition records: " + str(data.shape[0]))

data = data[['location', 'date', 'total_cases_per_million', 'total_deaths_per_million',
    'population_density', 'stringency_index', 'reproduction_rate', 'gdp_per_capita',
    'median_age', 'human_development_index', 'hospital_beds_per_thousand', 'diabetes_prevalence',
    'days_after_100_cases', 'internal_flights', 'international_arrivals', 'new_cases']]

training_data = data[data['date'].str.contains("2020")]
testing_data = data[data['date'].str.contains("2021")]

#print(training_data)
training_data.to_csv('rf_training.csv')
testing_data.to_csv('rf_testing.csv')