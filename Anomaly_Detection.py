#This program is a class used to detect anomalous data

class anomalyDetection:   
    def __init__(self):
        pass
    def dataProcess(self):
        import pingouin as pg
        import pandas as pd
        import numpy as np
        import datetime
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import LabelEncoder

        #load the data in excel
        x=pd.read_csv('topline_metrics_Input.csv')
        #set the data set to be the format of data frame
        data_org=pd.DataFrame(data=x)
        #delete duplicated records
        data=data_org.drop_duplicates()
        #delete records with negative values for "Time per Day"
        positiveTime=data['Time Spend Per Day(seconds)']>0
        data=data[positiveTime]
        #encode the categorical factors
        label_encoder = LabelEncoder()
        data.Country = label_encoder.fit_transform(data.Country)
        data.Date = self.string2integer(data)
        data.Platform = label_encoder.fit_transform(data.Platform)
        #Apply isolation forest algorithm to do the anomaly detection
        clf=IsolationForest(random_state=0).fit(data)
        outlier=clf.predict(data)
        outlierLocations=np.where(outlier==-1)
        print(outlierLocations[0])
        #output the locations of observations which are identified as outliers
        self.outlierRows=outlierLocations[0]
        #delete the detected amolous records
        self.dataset=data.drop(data.index[self.outlierRows])
        self.outlierColumn=outlier
    # Build a function to preprocess "Date", convert "Date" from string to integer
    def string2integer(self,dtFm):
        from datetime import datetime
        for i,row in dtFm.iterrows():
            dateString=row['Date']
            dateTypeData=datetime.strptime(dateString,'%m/%d/%Y')
            dateInteger=dateTypeData.timestamp()
            dtFm.at[i,'Date']=dateInteger
        return dtFm

if __name__ == '__main__': 
    import pandas as pd
    aD=anomalyDetection()
    aD.dataProcess()
    