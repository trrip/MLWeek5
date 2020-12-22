import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib.dates import date2num
from geopandas import gpd
from shapely.geometry import Point
import matplotlib.colors as colors
#LANDSLIDE DATA
df = pd.read_csv("landslides-edit.csv")
#World data for map plot & drop columns
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = world.drop(columns=['continent', 'pop_est', 'gdp_md_est'])
#drop unused columns
df.drop(columns=['source_name', 'source_link', 'event_id', 'event_time', 'event_title','event_description', 'location_description',
                    'location_accuracy', 'landslide_setting', 'injury_count', 'storm_name', 'photo_link', 'notes', 'event_import_source', 'event_import_id',
                    'admin_division_name', 'gazeteer_closest_point', 'submitted_date', 'submitted_date', 'created_date', 'last_edited_date','landslide_category','landslide_trigger','landslide_size'
                    ,'fatality_count','admin_division_population','gazeteer_distance', 'longitude', 'longitude'],
                    axis=1, inplace= True)               
#remove any rows that have NaN
df.dropna(
    axis=0,
    how='any',
    thresh=None,
    subset=None,
    inplace=True
)
df.insert(1, 'landslides', 'event')

df['month_year'] = pd.to_datetime(df['event_date']).dt.to_period('Y')
#get differences in landslide occurance 
data = df.groupby(["month_year","country_name","landslides"],as_index=False).size()
landslides_in_2008_per_country = data.loc[data['month_year'] == '2008']
landslides_in_2016_per_country = data.loc[data['month_year'] == '2016']
inner_merge = pd.merge(landslides_in_2016_per_country, landslides_in_2008_per_country, on=['country_name'], how='inner')
differences_by_country = []
difference = 0
for index, row in inner_merge.iterrows():
    difference = row['size_x'] - row['size_y']
    differences_by_country.append(difference)
inner_merge['difference_from_08_16'] = differences_by_country
inner_merge = inner_merge.drop(columns=['month_year_x' , 'landslides_x', 'size_x', 'month_year_y', 'landslides_y' , 'size_y'])

#problem, some rows dropped
for_plotting = world.merge(inner_merge, left_on = 'name', right_on = 'country_name')
for_plotting = for_plotting.drop(columns = ['country_name'])
for_plotting = pd.merge(world, for_plotting, on = ['name'], how = 'outer')
for_plotting = for_plotting.drop(columns = ['name', 'iso_a3_y', 'geometry_y'])
for_plotting.columns = ['iso_a3', 'geometry', 'difference_from_08_16']
print(for_plotting)
ax = for_plotting.plot(column='difference_from_08_16', missing_kwds={'color': 'lightgrey'},cmap = 'OrRd', figsize=(15,9),  scheme='quantiles', k=20, legend = True, label = 'No Data');
ax.set_title('Total Increase or Decrease in Landslides Between 2008 and 2016', fontdict= {'fontsize':25})
plt.show()