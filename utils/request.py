"""

    Simple Script to test the API once deployed

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located at the root of this repo for guidance on how to use this
    script correctly.
    ----------------------------------------------------------------------

    Description: This file contains code used to formulate a POST request
    which can be used to develop/debug the Model API once it has been
    deployed. 

"""

# Import dependencies
import requests
import pandas as pd
import numpy as np

# Load data from file to send as an API POST request.
# We prepare a DataFrame with the public test set + riders data
# from the Zindi challenge.
test = pd.read_csv('data/test_data.csv')
riders = pd.read_csv('data/riders.csv')
test = test.merge(riders, how='left', on='Rider Id')

# Convert our DataFrame to a JSON string.
# This step is necessary in order to transmit our data via HTTP/S
feature_vector_json = test.iloc[1].to_json()

# Specify the URL at which the API will be hosted.
# NOTE: When testing your instance of the API replace the URL
# below with your own:

# url = 'http://{your-ip-address}:5000/api_v0.1'
url = 'http://127.0.0.1:5000/api_v0.1'

# Perform the POST request.
print (f"Querying API with the following data: {feature_vector_json}")
print ("")
# Here `api_response` represents the response we get from our API
api_response = requests.post(url, json=feature_vector_json)

# Display the prediction result
print("*"*50)
print(f"API prediction result: {api_response.json()}")
print(f"The response took: {api_response.elapsed.total_seconds()} seconds")
print("*"*50)
