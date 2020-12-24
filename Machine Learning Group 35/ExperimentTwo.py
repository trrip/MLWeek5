
#Library Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import pandas_profiling

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import kneighbors_graph
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.svm import SVC


# Data Import
df = pd.read_csv('landslides.csv' , comment="#")
df['event_date'] = pd.to_datetime(df['event_date'])
df['submitted_date'] = pd.to_datetime(df['submitted_date'])
df['created_date'] = pd.to_datetime(df['created_date'])
df['last_edited_date'] = pd.to_datetime(df['last_edited_date'])


# # Data Cleaning

# The unnecessary columns of the dataset are filtered from the featureset
# Null values are set to 0 or removed from the dataset.

featureset = df[["landslide_category","landslide_trigger","landslide_size", 
         "landslide_setting", "fatality_count", "injury_count"]]
featureset = featureset.fillna(0)

x = featureset[["landslide_category","landslide_trigger","landslide_size", "landslide_setting", "injury_count"]]
y = featureset["fatality_count"]

num_entries = len(df)
num_nans = featureset['injury_count'].count()
num_entries = len(df)


#df.profile_report()



x = x.astype(str)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 2)
x_test



#As we have categorical data, this function encodes each string category
# with a number to be used in various ML algorithms.
def prepare_inputs(X_train, X_test):
	oe = OrdinalEncoder()
	oe.fit(x)
	X_train_enc = oe.transform(X_train)
	X_test_enc = oe.transform(X_test)
	return X_train_enc, X_test_enc

X_train_enc, X_test_enc = prepare_inputs(x_train, x_test)
X_train_enc
#scaler = StandardScaler()
#scaled_data = scaler.fit_transform(x)


i = 0
for col in  x.columns:
    
    plt.figure(figsize=(18, 9))
    plt.rc("font", size=14)
    #plt.scatter(X_train_enc[:,1], y_pred, color="black")
    plt.scatter(x.iloc[:,i], y, color="black")
    plt.xlabel(col)
    plt.xticks(rotation=45)
    plt.ylabel("Number of Fatalities")
    plt.title("Number of Landslide Related Fatalities Per Category")
    i = i + 1


# Creating regression model to predict number of fatalities

lm = linear_model.LinearRegression(normalize = True)
model = lm.fit(X_train_enc,y_train)
y_pred = model.predict(X_test_enc)
print("Model Parameters are: \n")
i = 0
for col in x.columns:
    print(col, ": ", lm.coef_[i])
    i+=1



# Visualisation of fatality predictions across different variables
i = 0
for col in  x.columns:
    
    plt.figure(figsize=(18, 9))
    plt.rc("font", size=14)
    #plt.scatter(X_train_enc[:,1], y_pred, color="black")
    plt.scatter(X_test_enc[:,i], y_pred, color="red", marker = 'x', s= 60)
    plt.scatter(X_train_enc[:,i], y_train, color="black")
    plt.xlabel(col)
    plt.xticks(rotation=45)
    plt.ylabel("Number of Fatalities")
    plt.title("Number of Landslide Related Fatalities Per Category")
    i = i + 1



df['Number'] =1
df['Year'] = df['event_date'].dt.year
featureset_ = df.groupby(['Year'], as_index=False).sum()

from mlxtend.plotting import plot_decision_regions


x = df['Year']

fig = plt.figure()
plt.xlabel("Occurence Date") 
plt.ylabel("Frequency")
plt.title("Number of Recorded Landslides Across Time")
plt.scatter(featureset_['Year'], featureset_['Number'], label = "Raw Data")
#plt.plot(x, func(x, *popt), 'r-', label="Fitted Curve")
plt.legend()



#Categorising landslide size

#As we have categorical data, this function encodes each string category
# with a number to be used in various ML algorithms.

x = featureset[["landslide_category","landslide_trigger", "landslide_setting", "injury_count","fatality_count"]]

y = featureset["landslide_size"]
y = y.astype('category')
y = y.cat.codes

x = x.astype(str)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 2)

def prepare_inputs(X_train, X_test):
	oe = OrdinalEncoder()
	oe.fit(x)
	X_train_enc = oe.transform(X_train)
	X_test_enc = oe.transform(X_test)
	return X_train_enc, X_test_enc

X_train_enc, X_test_enc = prepare_inputs(x_train, x_test)
y_train


df1 = pd.DataFrame(data=y, columns=['Category'])
df1['count'] = 1
df1 = df1.groupby(['Category']).sum()
df1



# generate logistic regression model
C = 1
alpha = 1/ (2*C)
model = LogisticRegression(C= alpha)
# fit model to training set.
model.fit(X_train_enc,y_train)
#use model set to predict y values on test set.
y_pred=model.predict(X_test_enc)
print(model.coef_, model.intercept_)
coefs = model.coef_
inter = model.intercept_
# formula for decision boundary line.
y_pred



# CODE FOR CONFUSION MATRIX SOURCED FROM:
#https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
confusion_matrix

class_names=[-1, 1]
tick_marks = [-1, 1]
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="YlGnBu" ,fmt='g', 
            yticklabels=tick_marks, xticklabels = tick_marks)

plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average = 'weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred,average = 'weighted'))
y

#Categorising landslide size

#As we have categorical data, this function encodes each string category
# with a number to be used in various ML algorithms.

x = featureset[["landslide_category","landslide_trigger", "landslide_setting", "injury_count","fatality_count"]]

y = featureset["landslide_size"]
y = y.astype('category')
y = y.cat.codes

x = x.astype(str)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 2)


X_train_enc, X_test_enc = prepare_inputs(x_train, x_test)


# penalty paramter c is altered to visualise the effect on the model.
c = 1
# generate supprt vector classifier and fit to training data.
clf = svm.LinearSVC(C = c, max_iter=10000)
clf.fit(X_train_enc,y_train)
svm = LinearSVC(C = c)
svm.fit(X_train_enc, y_train)

# use trained model to predict y-values.
y_pred = svm.predict(X_test_enc)
print(svm.coef_)
print(svm.intercept_)

# CODE FOR CONFUSION MATRIX SOURCED FROM:
#https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
confusion_matrix

class_names=[-1, 1]
tick_marks = [-1, 1]
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="YlGnBu" ,fmt='g', 
            yticklabels=tick_marks, xticklabels = tick_marks)

plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average = 'weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred,average = 'weighted'))

