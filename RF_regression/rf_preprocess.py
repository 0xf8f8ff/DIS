import pandas as pd
import sys
from datetime import datetime as dt


data = pd.read_csv('updated_dataset.csv')


data = data[['location','date','total_deaths','new_deaths','new_deaths_smoothed','total_cases_per_million',
'total_deaths_per_million','new_deaths_smoothed_per_million','reproduction_rate','total_tests',
'stringency_index','population','population_density','median_age','aged_65_older','aged_70_older',
'gdp_per_capita','extreme_poverty','cardiovasc_death_rate','diabetes_prevalence','female_smokers',
'male_smokers','hospital_beds_per_thousand','life_expectancy','human_development_index','new_tests',
'days_after_100_cases','internal_flights','international_arrivals','new_cases']]

data.fillna(0)


training_data = data[data['date'].str.contains("2020")]
testing_data = data[data['date'].str.contains("2021")]

#print(training_data)
training_data.to_csv('rf_training.csv')
testing_data.to_csv('rf_testing.csv')