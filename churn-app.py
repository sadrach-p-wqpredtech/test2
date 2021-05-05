#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:47:57 2021

@author: sadrachpierre
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



st.write("""
# Churn Prediction App
Customer churn is defined as the loss of customers after a certain period of time. Companies are interested in targeting customers
who are likely to churn. They can target these customers with special deals and promotions to influence them to stay with
the company. 

This app predicts the probability of a customer churning using Telco Cusstomer data. Here
customer churn means the customer does not make another purchase after a period of time. 

""")

st.sidebar.header('User Input Features')


df_selected = pd.read_csv("telco_churn.csv")
df_selected['target'] = np.where(df_selected['Churn']=='Yes', 1, 0)
df_selected['gender'] = np.where(df_selected['gender']=='Female', 1, 0)
df_selected['Partner'] = np.where(df_selected['Partner']=='Yes', 1, 0)
df_selected['Dependents'] = np.where(df_selected['Dependents']=='Yes', 1, 0)
df_selected['PhoneService'] = np.where(df_selected['PhoneService']=='Yes', 1, 0)

df_selected_all = df_selected[['gender', 'Partner', 'Dependents', 'PhoneService', 
                                     'tenure', 'MonthlyCharges', 'target']].copy()

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="churn_data.csv">Download CSV File</a>'
    return href

st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown(filedownload(df_selected_all), unsafe_allow_html=True)

# Heatmap
if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    st.write("""Correlation heatmaps are a great way to visualize, not only the relationship betwen input variables, but also the relationship
     between our inputs and our target. This can help with identifying which input features most strongly influence an outcome. In our heatmap
     we see that there is a strong negative relationship between tenure and churn and a strong positive relationship between MonthlyCharges and churn.""")
    df_selected_all.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')

    corr = df.corr()
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, cmap="Blues", annot=True)
    st.pyplot()
    


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        gender = st.sidebar.selectbox('gender',('Male','Female'))
        PaymentMethod = st.sidebar.selectbox('PaymentMethod',('Bank transfer (automatic)', 'Credit card (automatic)', 'Mailed check', 'Electronic check'))
        MonthlyCharges = st.sidebar.slider('Monthly Charges', 32.1,59.6,43.9)
        tenure = st.sidebar.slider('tenure', 13.1,21.5,17.2)

        data = {'gender':[gender], 
                'PaymentMethod':[PaymentMethod], 
                'MonthlyCharges':[MonthlyCharges], 
                'tenure':[tenure],}
        
        features = pd.DataFrame(data)
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
churn_raw = pd.read_csv('telco_churn.csv')




churn_raw.fillna(0, inplace=True)
churn = churn_raw.drop(columns=['Churn'])
df = pd.concat([input_df,churn],axis=0)



# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['gender','PaymentMethod']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)
df.fillna(0, inplace=True)


features = ['MonthlyCharges', 'tenure', 'gender_Female', 'gender_Male',
       'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

df = df[features]



# Displays the user input features
st.subheader('User Input features')
print(df.columns)
if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('churn_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
churn_labels = np.array(['No','Yes'])
st.write(churn_labels[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
