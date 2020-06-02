"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------

    feature_vector_df.columns = [name.lower() for name in feature_vector_df.columns]
    #feature_vector_df.drop(["arrival at destination - day of month","arrival at destination - time"],axis=1,inplace=True)

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
        df["personal or business"] = pd.get_dummies(df, prefix =["personal or business"],columns = ["personal or business"],drop_first=True)
        df['time from arrival to pickup'] = 0
        return df

    feature_vector_df = correct_data_format_train(feature_vector_df)

    predict_vector = feature_vector_df
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
