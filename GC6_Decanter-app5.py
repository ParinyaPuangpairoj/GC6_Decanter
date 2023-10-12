import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.express as px

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
st.image('Decanter GC6-image.png')

st.sidebar.header('User Input Features for Real Condition')

st.sidebar.markdown("""
[Example CSV input file]-->
Date,	T-5551,	T-5554,	T-5552,	Decanter Flowrate,	Temp bearing,	Temp bearing,	Vibration (Feed),	Vibration (Driver),	Vibration (Frame),	Blow speed (RPM),	Diff speed (RPM),	Torque (%),	Severity

""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

st.sidebar.header('User Input Features Simulation')

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

# Selects the all rows (the user input data)
df = df[:] 

# Reads in saved classification model
load_clf = pickle.load(open('GC6_Decanter_rfr.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)

#---------------------------------------------------------------------
# export to csv file 
@st.cache_data
def convert_df(df9):
    # Important: Cache the conversion to prevent computation on every rerun
    return df9.to_csv().encode('utf-8')

#---------------------------------------------------------
if uploaded_file is None:
        # Displays the user input features
        st.subheader('User Input features Simulation')
        st.write(df)
        
        st.subheader('Simulation and Prediction')
        st.write([prediction])
        
else:
        predict1 = pd.Series(prediction, name='Severity')
        df_full_prediction = pd.concat([input_df2.iloc[:,:13], pd.Series(predict1)], axis=1)
        
        # Displays the user input features
        st.subheader('User Input Features for Real Condition')
        st.write('Awaiting CSV file to be uploaded. Currently using input parameters (shown below).')
        st.write(input_df2)
        st.subheader('Prediction')
        st.write(df_full_prediction)
        csv1 = convert_df(df_full_prediction)

        # Line Chart plot for Severity Prediction
        fig = px.line(df_full_prediction, x="Date", y="Severity", title='The Decanter health prediction')
        fig.show()

        # Download CSV file
        st.download_button(
                label = "Download the Decanter health prediction as csv file",
                data = csv1,
                file_name = 'Decanter_predictive_file.csv',
                mime = 'text/csv'
                                )


