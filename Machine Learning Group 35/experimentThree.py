import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.axes as axes
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib.dates import date2num
import matplotlib.dates as mdates
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("landslides.csv")

#drop unused columns
df.drop(columns=['source_name', 'source_link', 'event_id', 'event_time', 'event_title','event_description', 'location_description',
                    'location_accuracy', 'landslide_setting', 'injury_count', 'storm_name', 'photo_link', 'notes', 'event_import_source', 'event_import_id',
                    'country_name', 'country_code', 'admin_division_name', 'gazeteer_closest_point', 'submitted_date', 'submitted_date', 'created_date', 'last_edited_date',],
                    axis=1, inplace= True)

#remove any rows that have NaN
df.dropna(
    axis=0,
    how='any',
    thresh=None,
    subset=None,
    inplace=True
)

df.drop(columns=['landslide_category', 'landslide_trigger', 'landslide_size', 'fatality_count', 
    'admin_division_population', 'gazeteer_distance', 'longitude', 'latitude'],axis=1, inplace= True)


#print(df.dtypes)

df.insert(1, 'landslides', 'event')

df['month_year'] = pd.to_datetime(df['event_date']).dt.to_period('Y')
print(df.head())

data = df.groupby(["month_year","landslides"],as_index=False).size()
print(data)

# Boiler plate for Linear regression used from this example for conveniece: 
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

#insert years seperately as there is a tough type error bug
data.insert(1, 'years', [1988, 1993, 1995, 1996, 1997, 1998, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
    2010, 2011, 2012, 2013, 2014, 2015, 2016], allow_duplicates = False)
print(data)

x = data['years'].values.reshape(-1,1)
y = data['size']
print(df.dtypes)
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2)

# model training
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

y_pred = regr.predict(x)

#model score
print(regr.score(x_train, y_train)) 

print(regr.score(x_test, y_test))

print(regr.score(x,y))

# plotting
years = mdates.YearLocator()   # every year

plt.scatter(x, y,  color='black', label='Total Events per Year')
plt.plot(x, y_pred, color='blue', linewidth=3, label='Linear Regression Prediction')
plt.legend()
plt.title("Quantity of Unique Events per Year with Predcition")
print(data['size'])
plt.show()