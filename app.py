# create environment for windows
# python -m venv myenv
# activate environment
# myenv\Scripts\activate
# pip install streamlit scikit-learn pandas seaborn numpy
# streamlit run app.py
import pickle
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np


from sklearn.preprocessing import StandardScaler

# load model
model = pickle.load(open('rf_model.pkl','rb'))

# title for app
st.title("Heart Attack Risk Classification App ❤️")

# create input features

age = st.number_input('Age',min_value=20,max_value=100,value=25)
restingbp = st.number_input('RestingBP',min_value=0 , max_value =300,value=100)
cholesterol = st.number_input('Cholesterol',min_value=0 , max_value =700,value=140)
fastingbs = st.selectbox('FastingBS',(0,1))
maxhr = st.number_input('MaxHR',min_value=60 , max_value =250,value=140)
oldpeak = st.number_input('Oldpeak',min_value=-3.0 , max_value =6.6,value=1.0)
gender = st.selectbox('Gender(Male or Female)',('M','F'))
chestpaintype = st.selectbox('ChestPainType',('ATA', 'NAP' ,'ASY' ,'TA'))
restingecg = st.selectbox('RestingECG',('Normal' ,'ST' ,'LVH'))
exerciseangina = st.selectbox('ExerciseAngina',('N' ,'Y'))
st_slope = st.selectbox('ST_Slope',('Up', 'Flat', 'Down'))

# Encoding logic
# exerciseangina
Exercise_Angina = 1 if exerciseangina=='Y' else 0

# Sex
Sex_F = 1 if gender=='F' else 0
Sex_M = 1 if gender=='M' else 0

# Chest_PainType
Chest_PainType_dict = {'ASY':3,'NAP':2,'ATA':1,'TA':0}
Chest_PainType = Chest_PainType_dict[chestpaintype]

# Resting_ECG
Resting_ECG_dict = {'Normal':0,'LVH':1,'ST':2}
Resting_ECG = Resting_ECG_dict[restingecg]

# ST_Slope
st_Slope_dict = {'Down':0,'Up':1,'Flat':2}
st_Slope = st_Slope_dict[st_slope]

# create dataframe
input_features = pd.DataFrame({
    'Age':[age],
    'RestingBP':[restingbp],
    'Cholesterol':[cholesterol],
    'FastingBS':[fastingbs],
    'MaxHR':[maxhr],
    'Oldpeak':[oldpeak],
    'Exercise_Angina':[Exercise_Angina],
    'Sex_F':[Sex_F],
    'Sex_M':[Sex_M],
    'Chest_PainType':[Chest_PainType],
    'Resting_ECG':[Resting_ECG],
    'st_Slope':[st_Slope]
})
# scaling
scaler = StandardScaler()
input_features[['Age','RestingBP','Cholesterol','MaxHR']]=scaler.fit_transform(input_features[['Age','RestingBP','Cholesterol','MaxHR']])

# predictions
if st.button('Predict'):
  predictions= model.predict(input_features)[0]
  if predictions==1:
    st.error('⚠️High Risk of Heart attack❗')
  else:
    st.success('Low risk of Heart attack😎😊')
