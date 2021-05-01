import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

def toDate(datestrings):
    format = '%Y-%m-%d'
    dates = []
    for dstr in datestrings:
        date = datetime.datetime.strptime(dstr, format)
        dates.append(date)
    return dates

training = pd.read_csv("rf_training.csv")
testing = pd.read_csv("rf_testing.csv")

training.fillna(0)
testing.fillna(0)

training = training.to_numpy()
testing = testing.to_numpy()
training = np.nan_to_num(training, 0)
testing = np.nan_to_num(testing, 0)

values = training[:, -1]
training = np.delete(training, obj = [0, 1, 2, -1], axis = 1).astype("float32")
training = np.nan_to_num(training, 0)
testing = np.nan_to_num(testing, 0)
values = np.nan_to_num(values, 0)

rf = Pipeline([('scaler2', StandardScaler()),
                  ('RandomForestRegressor: ', RandomForestRegressor(n_estimators=1000))])

rf.fit(training, values)

actual = testing[:, [1, 2, -1]]
testing = np.delete(testing, obj = [0, 1, 2, -1], axis = 1).astype("float32")
testing = np.nan_to_num(testing, 0)
predictions = rf.predict(testing)
predictions = np.atleast_2d(predictions).T.astype("int")

results = np.append(actual, np.array(predictions), 1)

df = pd.DataFrame(results, columns=["location", "date", "actual", "predicted"])

countries = [
    'India', 'China', 'Iran', 'South Korea', 'South Africa', 'Kenya',
    'Bangladesh', 'Sweden', 'Norway', 'Germany', 'Italy', 'United Kingdom',
    'Brazil', 'United States', 'Canada', 'Australia'
]

mr = pd.read_csv("rfresults.csv")
spark = pd.read_csv("spark_predictions")


fig, axes = plt.subplots(7, 2, figsize=(8,12))
fig.suptitle("Predictions for 2021")
i = 0
dateaxis = []
ldates = []
labels = ["predicted", "actual", "spark", "sklearn"]

plots = []

for x, ax in enumerate(fig.axes):
    
    local = mr.loc[mr["location"] == countries[i]]
    sk = df.loc[df["location"] == countries[i]]
    sp = spark.loc[spark['location'] == countries[i]]
    if len(local) != 0 and i < 15: 
        ax.set_title(countries[i])
        dateaxis = toDate(local["date"])
        xlabels = dateaxis[::20]
        xlabels.insert(0, dateaxis[0])
        xlabels.insert(-1, dateaxis[-1])
        xdates = local["date"][::20].tolist()
        xdates.insert(0, xdates[0])
        xdates.insert(-1, xdates[-1])
        ax.set_xticks(xlabels)
        ax.set_xticklabels(xdates, fontsize =  'xx-small')
        
        l1 = ax.plot(dateaxis, local["predicted"], label="predicted")
        l2 = ax.plot(dateaxis, local["actual"], label="actual")
        l3 = ax.plot(dateaxis, sp["prediction"], label="spark")
        l4 = ax.plot(dateaxis, sk["predicted"], label="sklearn")
        if i == 0:
            plots = [l1, l2, l3, l4]
        ax.xaxis_date()
        
        i += 1
fig.legend(plots, labels=labels, loc="upper right")
plt.show()