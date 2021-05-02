import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA

def dataPreprocessKMeans():

    url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
    df = pd.read_csv(url, sep='\,', na_values='.', engine='python')
    # df = pd.read_csv('updated_dataset.csv')

    url2 = "https://datagraphics.dckube.scilifelab.se/api/dataset/bbbaf64a25a1452287a8630503f07418.csv"
    df_tests_Sweden = pd.read_csv(url2, sep='\,', na_values='.', engine='python')

    country_list = df['location'].unique()
    countries = country_list.tolist()
    df = df[df['location'].isin(countries)]
    df = df.reset_index(drop=True)

    df_countries = df.copy()

    df = df.drop([
        'iso_code', 'continent', 'new_cases_per_million', 'new_deaths_per_million',
        'icu_patients_per_million', 'hosp_patients_per_million',
        'weekly_icu_admissions', 'weekly_icu_admissions_per_million',
        'weekly_hosp_admissions', 'weekly_hosp_admissions_per_million',
        'new_tests', 'total_tests_per_thousand', 'new_tests_per_thousand',
        'new_tests_smoothed', 'new_tests_smoothed_per_thousand', 'tests_units',
        'handwashing_facilities', 'hosp_patients', 'icu_patients', 'positive_rate',
        'tests_per_case','reproduction_rate'
    ],
                 axis=1)

    df.loc[df.location == 'Germany', 'extreme_poverty'] = 0.2

    df3 = pd.DataFrame(columns=df.columns)

    for location in countries:#df.location.unique():
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
        for column_name in ['total_deaths']:#, 'reproduction_rate']:
            index_of_first_valid = df_location[column_name].first_valid_index()
            if index_of_first_valid == None:
                continue
            df_location.loc[df_location.index < index_of_first_valid, column_name] = 0

        df_location = df_location.reset_index(drop=True)

        # Creates column based on index of dataframe
        df_location['days_after_100_cases'] = df_location.index
        # Appending the new dataframes df_location to df3
        df3 = df3.append(df_location, ignore_index=True)

    df = df3

    # Applying actual data to 'smoothed' columns where value is NaN
    df.loc[df.new_cases_smoothed_per_million.isna(), 
           'new_cases_smoothed_per_million'] = df.loc[:,'new_cases'] * 1000000 / df.loc[:,'population']

    df.loc[df.new_deaths_smoothed_per_million.isna(), 
           'new_deaths_smoothed_per_million'] = df.loc[:,'new_deaths'] * 1000000 / df.loc[:,'population']

    df.loc[df.new_cases_smoothed.isna(),
           'new_cases_smoothed'] = df.loc[:, 'new_cases']
    df.loc[df.new_deaths_smoothed.isna(),
           'new_deaths_smoothed'] = df.loc[:, 'new_deaths']

    # Takes the last valid value for 'stringency_index' and applies it to the last days of the dataset 
    #for United Kingdom and Brazil.
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
    df = df.fillna(0)

    columns = df.columns

    columns = columns.tolist()

    df_grouped = df.groupby('location')
    taking_maximums = df_grouped.max()
    final_df = taking_maximums.reset_index()
    locations = final_df['location']

    final_df = final_df.drop(['location','date'],axis=1)

    # final_df.insert(0, 'cluster', np.random.randint(1, 4, final_df.shape[0]))
    # final_df.insert(0, 'id', range(1, 1 + len(final_df)))
    print(final_df.shape)

    matDF = final_df.to_numpy()
    np.savetxt('orgdata.csv', matDF, delimiter=',')
    np.savetxt('orgdata.txt', matDF, delimiter=',')

    # normalized_matDF = preprocessing.normalize(matDF)
    # standardisation = preprocessing.StandardScaler() 
    # standardised_matDF = standardisation.fit_transform(normalized_matDF)

    # Direct PCA using builtin PCA
    pca = PCA(n_components=3)
    pca_DF = pca.fit_transform(matDF)
    # pca_DF = pca.fit_transform(standardised_matDF)

    # cl = np.random.randint(1, 4, final_df.shape[0])
    # idd = list(range(1, 1 + len(final_df)))
    # idd = np.array(idd)#.to_numpy()

    clusterss = np.random.randint(1, 8, final_df.shape[0])
    ids = list(range(1, 1 + len(final_df)))
    ids = np.array(ids)#.to_numpy()
    ids = ids.tolist()

    # Final data with ID and tagged with random clusters (used the data after doing builtin PCA)
    cleaned_data = np.vstack((clusterss, pca_DF.T)).T
    cleaned_data = np.vstack((ids, cleaned_data.T)).T
    np.savetxt('testdata.txt', cleaned_data)

    # this data has been prepared for mapreduce PCA
    # normalized_matDF = preprocessing.normalize(matDF)
    # standardisation = preprocessing.StandardScaler()
    # standardised_matDF = standardisation.fit_transform(normalized_matDF)
    data_for_mapreduce_pca = np.vstack((ids, matDF.T)).T
    np.savetxt('pcaorgdata.txt', data_for_mapreduce_pca)
    print((data_for_mapreduce_pca.size))
    
    return None


if __name__ == "__main__":
    dataPreprocessKMeans()