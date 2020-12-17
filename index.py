from sys import stderr
from numpy.core.fromnumeric import mean
from numpy.lib.type_check import nan_to_num
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
FOLD_CONSTANT = 5


def cleanData(list):
    x_new = {}
    counter = 0.01
    return_list = []
    finalList = []
    for i in list:
        if i == "":
            return_list.append(0)
            continue
        if i in x_new:
            return_list.append(x_new[i])
        else:
            x_new[i] = counter
            counter += 0.01
            return_list.append(x_new[i])
    for i in return_list:
        finalList.append(nan_to_num(i, nan=0.0))
    return finalList


def checkForNAN(list):
    returnList = []
    for i in list:
        try:
            int(i)
        except Exception:
            print("something wrong with this ")
            print(i)


def cleanNumberData(list):
    return_list = []
    for i in list:
        return_list.append(nan_to_num(i, nan=0.0))
    return return_list


def getData(fileName):
    df = pd.read_csv(
        fileName,
        comment="#",
    )
    # x0 = df.iloc[:, 0]  # event id dont use
    # x1 = df.iloc[:, 1]  # event date dont use
    x2 = df.iloc[:, 2]  # landslidecategory
    x3 = df.iloc[:, 3]  # landslide_trigger
    x4 = df.iloc[:, 4]  # landslide_size
    x5 = df.iloc[:, 5]  # admin_division_population

    # x6 = df.iloc[:, 6]
    # x7 = df.iloc[:, 7]
    # x8 = df.iloc[:, 8]
    x2 = np.array(cleanData(x2))
    x3 = np.array(cleanData(x3))
    x4 = np.array(cleanData(x4))
    x5 = np.array(cleanData(x5))
    x = np.column_stack((x2, x3, x4, x5))
    y = cleanNumberData(df.iloc[:, 9])
    # y = df.iloc[:, 9]
    return x, y


def linearSolution(x, y, c=10):
    cValues = [0.1, 0.2, 0.33, 0.4]  #, 1, 10, 25, 100, 200, 400, 500]
    kf = KFold(n_splits=5)
    stdError = []
    meanError = []
    numberOfZero = []
    cValue = []
    score = []
    for c in cValues:
        model = LinearRegression()
        X_train, X_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=c,
                                                            random_state=42)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        model.fit(X_train, y_train)
        cValue.append(c)
        ytestingData = model.predict(X_test)
        meanError.append(mean_squared_error(y_test, ytestingData))
    plt.plot(meanError, cValues)
    plt.xlabel("Values of Folds")
    plt.title(f"with mean square error For")
    plt.legend(["variance"])
    plt.ylabel("mean square error")
    plt.show()


def lassoSolution(x, y):
    cValues = [0.1, 1, 10, 50]  #, 1, 10, 25, 100, 200, 400, 500]
    kf = KFold(n_splits=5)
    stdError = []
    meanError = []
    numberOfZero = []
    cValue = []
    score = []
    for c in cValues:
        model = Lasso(alpha=(1 / 1 + c))
        X_train, X_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.25,
                                                            random_state=42)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        model.fit(X_train, y_train)
        cValue.append(c)
        ytestingData = model.predict(X_test)
        meanError.append(mean_squared_error(y_test, ytestingData))
    plt.plot(cValue, meanError)
    plt.xlabel("Values of Folds")
    plt.title(f"with mean square error For")
    plt.legend(["variance"])
    plt.ylabel("mean square error")
    plt.show()


if __name__ == "__main__":
    x, y = getData("dataset.csv")
    linearSolution(x, y)
    lassoSolution(x, y)
