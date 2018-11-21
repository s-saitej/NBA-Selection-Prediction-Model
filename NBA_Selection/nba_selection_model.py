import pandas as pd
import numpy as np

dataset = pd.read_csv('nba_logreg.csv') 

dataset.isnull().any()

dataset = dataset.fillna(0.00)

dataset.columns

dataset = dataset[['GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3P Made', '3PA',
       '3P%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK',
       'TOV', 'TARGET_5Yrs']]

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1:]

from sklearn import model_selection 

train_data,test_data,train_target,test_target = model_selection.train_test_split(x,y)

from sklearn import preprocessing

train_data = preprocessing.normalize(train_data)
train_target = preprocessing.normalize(train_target)
test_data = preprocessing.normalize(test_data)
test_target = preprocessing.normalize(test_target)

from sklearn import linear_model

regression = linear_model.LogisticRegression()
fitting = regression.fit(train_data,train_target)
result = regression.predict(test_data)

result

from matplotlib import pyplot
pyplot.hist(result)
pyplot.hist(test_target, color = 'orange')

pyplot.scatter(result,test_target)

cof = regression.coef_
intercept = regression.intercept_

from sklearn import metrics

varience = metrics.r2_score(result,test_target)
varience

mean_error = metrics.mean_squared_error(result,test_target)
mean_error

pyplot.scatter(result,test_target)
