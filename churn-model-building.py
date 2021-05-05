#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:44:59 2021

@author: sadrachpierre
"""
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import pickle


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df_churn = pd.read_csv('telco_churn.csv')
print(df_churn.head())

df_churn = df_churn[['gender', 'PaymentMethod', 'MonthlyCharges', 'tenure', 'Contract', 'Churn']].copy()

print(set(df_churn['Contract']))
print(df_churn.head())

df = df_churn.copy()
df.fillna(0, inplace=True)
target = 'Churn'
encode = ['gender','PaymentMethod', 'Contract']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper = {'No':0, 'Yes':1,}
def target_encode(val):
    return target_mapper[val]

df['Churn'] = df['Churn'].apply(target_encode)

# Separating X and y

X = df.drop('Churn', axis=1)
Y = df['Churn']



print(X.columns)
# Build random forest model

clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model

pickle.dump(clf, open('churn_clf.pkl', 'wb'))