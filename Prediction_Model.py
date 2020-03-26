# This is program is to do feature selection, then build prediction model between Y and impordation features

import pingouin as pg
import pandas as pd
import numpy as np
import datetime
from sklearn import preprocessing
from Anomaly_Detection import anomalyDetection 

#drop the detected anomalous obervations
aDt=anomalyDetection()
aDt.dataProcess()
data=aDt.dataset
X_sub=data.drop(columns={'Items per DAU'})
# Interested in study what factors have effect on "TIme spend per day"
Y_sub=data[["Items per DAU"]]
#standardize the data
from sklearn.preprocessing import StandardScaler
scaler_X=StandardScaler().fit(X_sub)
scaler_Y=StandardScaler().fit(Y_sub)
X_scaled=scaler_X.transform(X_sub)
Y_scaled=scaler_Y.transform(Y_sub)
#apply ExtraTreesRegressor to do feature selection
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X_scaled,Y_scaled)
print(model.feature_importances_) 
feat_importances = pd.Series(model.feature_importances_, index=X_sub.columns)
#Extra the important factors based on the feature importance
xBoolean=feat_importances.isin(feat_importances.nlargest(5))
X_select=X_scaled[:,xBoolean]
selectHead=X_sub.columns[xBoolean]
print(selectHead)

#use MLP to model X_select vs Y
from sklearn.neural_network import MLPRegressor
estimator=MLPRegressor()
mdl=estimator.fit(X_select,Y_scaled)
print(mdl.score(X_select,Y_scaled))    
