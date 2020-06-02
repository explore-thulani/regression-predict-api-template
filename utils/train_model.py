"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

# Fetch training data and preprocess for modeling
train = pd.read_csv('./data/train_data.csv')
riders = pd.read_csv('./data/riders.csv')

riders['RidersID'] = range(len(riders))
train = pd.merge(train,riders,on="Rider Id")
train.columns = [name.lower() for name in train.columns]

def remove_colon(number):
        if number[-1]==':' or number[-1]=='A' or number[-1]=='P':
            number = number[:-1]
        return number

def convert_dates_to_seconds(df, date,time):
    """takes in pandas dataframe and converts time stamps
    into seconds """
    date_in_seconds = []
    days_in_seconds = (df[date]-1)*3600*24
    
    for i in range(len(df)):
        hours = int(remove_colon(df[time][i][:2]))
        mins =  int(remove_colon(df[time][i][3:5]))
        seconds =  int(remove_colon(df[time][i][6:9]))
        
        if df[time][i][-2:] =='PM':
            date_in_seconds.append((hours+12)*3600+mins*60+seconds)
            
        else:
            date_in_seconds.append((hours)*3600+mins*60+seconds)
                
        
    
    return np.array(date_in_seconds)+np.array(days_in_seconds)

def correct_data_format_train(df):

    df = df.copy()
    """takes in data frame and transforms it into preferred data frame """
    
    #convert dates
    
    convert_columns = {"placement - day of month":"placement - time","confirmation - day of month":"confirmation - time",
                    "pickup - day of month":"pickup - time","arrival at pickup - day of month":"arrival at pickup - time"}
    
    drop_columns = ["precipitation in millimeters",
                    "arrival at destination - day of month",
            "arrival at destination - time","arrival at destination - weekday (mo = 1)","order no","user id","vehicle type","order no","rider id"]

    for column_name in df.columns:
        if column_name in convert_columns:
            df[convert_columns[column_name]] = convert_dates_to_seconds(df,date=column_name ,time =convert_columns[column_name])
        elif column_name in drop_columns:
            df.drop(column_name,axis=1,inplace=True)
            
    #Fillingna and dummy variable
    df['temperature'].fillna(df['temperature'].mean(),inplace=True)
    
    return df

train = correct_data_format_train(train)

train["personal or business"] = pd.get_dummies(train["personal or business"], prefix =["personal or business"],columns = ["personal or business"],drop_first=True)
print(train.columns)
x_train = train.drop("time from pickup to arrival",axis = 1).values
y_train = train.pop("time from pickup to arrival").values

# Fit model
RF = RandomForestRegressor(n_estimators=350, max_depth=15)
RF.fit(x_train,y_train)

# Pickle model for use within our API
save_path = '../assets/trained-models/sendy_random_forest_regression.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(RF, open(save_path,'wb'))
