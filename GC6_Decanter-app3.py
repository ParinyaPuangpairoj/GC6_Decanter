-m pip install --upgrade pip
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


st.write("""
# Decanter Prediction App

This app prediction the **Decanter** healthy!

Data obtained from the [GC6] in XX by MR PARINYA PU.
""")

st.sidebar.header('User Input Features Simulation')

st.sidebar.markdown("""
[Example CSV input file]
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])


if uploaded_file is not None:
    input_df2 = pd.read_csv(uploaded_file)
    input_df = input_df2.drop(columns=['Date','Severity'])
else:
    def user_input_features():
        T_5551 = st.sidebar.slider('T-5551', 0, 100, 50)
        T_5554 = st.sidebar.slider('T-5554', 0, 100, 50)
        T_5552 = st.sidebar.slider('T-5552', 0, 100, 50)
        Decanter_Flowrate = st.sidebar.slider('Decanter_Flowrate', 0, 20,10)
        Temp_Bearing_NDE = st.sidebar.slider('Temp_Bearing_NDE', 0,200,100)
        Temp_Bearing_DE = st.sidebar.slider('Temp_Bearing_DE', 0,200,100)
        Vibration_Feed = st.sidebar.slider('Vibration_Feed', 0 ,30, 15)
        Vibration_Driver = st.sidebar.slider('Vibration_Driver', 0 ,30, 15)
        Vibration_Frame = st.sidebar.slider('Vibration_Frame', 0 ,30, 15)
        Blow_Speed_RPM = st.sidebar.slider('Blow_Speed_RPM', -10, 4000, 2000 )
        Diff_Speed_RPM = st.sidebar.slider('Diff_Speed_RPM', -10, 20, 10)
        Torque = st.sidebar.slider('Torque', 0, 40, 20)
        data = {
                'T_5551': T_5551,
                'T_5554': T_5554,
                'T_5552': T_5552,
                'Decanter_Flowrate': Decanter_Flowrate,
                'Temp_Bearing_NDE': Temp_Bearing_NDE,
                'Temp_Bearing_DE': Temp_Bearing_DE,
                'Vibration_Feed' : Vibration_Feed,
                'Vibration_Driver' : Vibration_Driver,
                'Vibration_Frame' : Vibration_Frame,
                'Blow_Speed_RPM' : Blow_Speed_RPM,
                'Diff_Speed_RPM' : Diff_Speed_RPM,
                'Torque': Torque
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire decanter dataset

df = pd.concat([input_df],axis=0)

# Selects only the first row (the user input data)
df = df[:1] 

# Displays the user input features
st.subheader('User Input features Simulation')

if uploaded_file is not None:
   st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('GC6_Decanter_rfr.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)

#----------------------------------------------------------

st.subheader('Simulation and Prediction')
st.write([prediction])

#-----------------------------------------------------------

