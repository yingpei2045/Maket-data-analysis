
#This program is to check if data satisfy the assumption for statistical methods for anomaly detection
import pingouin as pg
import pandas as pd
import numpy as np
#load the data in excel
x=pd.read_csv('topline_metrics_Input.csv')
dataFrame=pd.DataFrame(data=x)
skewValue=dataFrame.skew(axis=0)
correlationMatrix=dataFrame.corr()
print(skewValue)
print(correlationMatrix)


