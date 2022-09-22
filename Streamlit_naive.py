import pandas as pd
import streamlit as st 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split # trian and test
from sklearn import metrics
from sklearn import preprocessing 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

st.title('Model Deployment: Naive Bayes classifier')

st.sidebar.header('User Input Parameters')

def user_input_features():
    industrial_risk = st.sidebar.selectbox('industrial_risk',('1','0','0.5'))
    management_risk = st.sidebar.selectbox('management_risk',('1','0','0.5'))
    financial_risk = st.sidebar.selectbox('financial_risk',('1','0','0.5'))
    credibility = st.sidebar.selectbox('credibilty',('1','0','0.5'))
    competitiveness= st.sidebar.selectbox('competitiveness',('1','0','0.5'))
    operating_risk= st.sidebar.selectbox('operating_risk',('1','0','0.5'))
    data = {'industrial_risk':industrial_risk,
            'management_risk':management_risk,
            'financial_risk':financial_risk,
            ' credibility': credibility,
            'competitiveness':competitiveness,
           'operating_risk ':operating_risk }
        
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

bankrupt = pd.read_csv("file:///D:/DATA SCIENCE/ExcelR/Live Projects/Bankrupt prevention/bankruptcy-prevention.csv", sep = ';', header = 0)

bankrupt_new = bankrupt.iloc[:,:]

bankrupt_new["class_yn"] = 1

bankrupt_new.loc[bankrupt[' class'] == 'bankruptcy', 'class_yn'] = 0

bankrupt_new.drop(' class', inplace = True, axis =1)

# Input
x = bankrupt_new.iloc[:,:-1]

# Target variable

y = bankrupt_new.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state = 0)
GNB = GaussianNB()
Naive_GNB = GNB.fit(x_train ,y_train)


prediction = GNB.predict(df)
prediction_proba = GNB.predict_proba(df)

st.subheader('Predicted Result')
st.write('Non Bankruptcy' if prediction_proba[0][1] > 0.5 else 'Bankruptcy')

st.subheader('prediction probability ')
st.write(prediction_proba)
