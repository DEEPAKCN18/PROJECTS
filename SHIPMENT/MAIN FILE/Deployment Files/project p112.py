# -*- coding: utf-8 -*-
import  pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import streamlit as st
import xgboost as xgb
#Backend part Starts
#Load dataset
df =pd.read_csv("C:\\Users\\hp\Downloads\\shipments.csv")
#removes unneccessary columns which doesn't give any idea
# Drop ID column
df = df.drop('ID', axis=1)
#Convert categorical columns in numerical and find unique values
Unique_disct=[]
for col in df:
    if (df.dtypes[col]==object):
        #Convert categorical col to Numeric
        codes, uniques = pd.factorize(df[col], sort=True)    #Create Dynamic dictionary to store unique values
        Unique_code=np.unique(codes)
        Unique_uniques=uniques.unique()
        #Dynamic Varianle Created
        globals()[f"Unique_dict{col}"] = dict(zip(Unique_code,Unique_uniques))
        df[col]=df[col].factorize(sort=True)[0]  #Convert categorical feature into numerical
# Split df into X and y
y = df['Reached.on.Time_Y.N']
X = df.drop('Reached.on.Time_Y.N', axis=1)
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
 # create a classifier object
xgb_cl = xgb.XGBClassifier()
# fit the classifier with X and Y data
xgb_cl.fit(X_train, y_train)
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.naive_bayes import GaussianNB
# create a classifier object
gnb = GaussianNB()
# fit the classifier with X and Y data
gnb.fit(X_train, y_train)
from sklearn.preprocessing import StandardScaler
array = X.values
scaler = StandardScaler().fit(array)
rescaledX = scaler.transform(array)
X = rescaledX
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.neighbors import KNeighborsClassifier
# create a classifier object
knn = KNeighborsClassifier(n_neighbors=5)
# fit the classifier with X and Y data
knn.fit(X_train, y_train)
#backend Part Ends
st.title("Model Deployment :customer analytics")
st.sidebar.header("User Input Parameters")
def user_input_features():
    Warehouse_block=st.sidebar.selectbox("Warehouse_block:",('A', 'B' ,'C', 'D', 'E','F'))
    Mode_of_Shipment=st.sidebar.selectbox("Mode_of_Shipment:",('Flight', 'Ship' ,'Road'))
    Customer_care_calls=st.sidebar.number_input(" Customer_care_calls(Ex.1):")
    Customer_rating=st.sidebar.selectbox("Customer_rating:", [2,5,3,1,4])
    Cost_of_the_Product=st.sidebar.number_input("Insert Cost_of_the_Product :")
    Prior_purchases=st.sidebar.number_input("Prior_purchase(ex.numbers from 1 t0 10):")
    Product_importance=st.sidebar.selectbox("Product_importance:",('low', 'medium' ,'high'))
    Gender=st.sidebar.selectbox("Gender:", ('F' 'M'))
    Discount_offered=st.sidebar.number_input("Discount_offered:")
    Weight_in_gms=st.sidebar.number_input("Weight_in_gms:")
    data={
        'Warehouse_block':Warehouse_block,
        'Mode_of_Shipment':Mode_of_Shipment,
        'Customer_care_calls':Customer_care_calls,
        'Customer_rating':Customer_rating,
        'Cost_of_the_Product':Cost_of_the_Product,
        'Prior_purchases': Prior_purchases,
        'Product_importance':Product_importance ,
        'Gender':Gender,
        'Discount_offered':Discount_offered,
        'Weight_in_gms':Weight_in_gms,        
        }
    features=pd.DataFrame(data,index=[0])
    return features

temp=user_input_features()
#print(df)
st.subheader("User Input Parameter")
st.write(temp)

#Convert Categorical values in Numeric
for key, value in Unique_dictWarehouse_block.items():
    temp['Warehouse_block'] = np.where(temp['Warehouse_block'] == value, key, temp['Warehouse_block'])
    temp['Warehouse_block']=temp['Warehouse_block'].factorize(sort=True)[0]
    
for key, value in Unique_dictMode_of_Shipment.items():
    temp['Mode_of_Shipment'] = np.where(temp['Mode_of_Shipment'] == value, key, temp['Mode_of_Shipment'])
    temp['Mode_of_Shipment']=temp['Mode_of_Shipment'].factorize(sort=True)[0]
     
for key, value in Unique_dictProduct_importance.items():
    temp['Product_importance'] = np.where(temp['Product_importance'] == value, key, temp['Product_importance'])
    temp['Product_importance']=temp['Product_importance'].factorize(sort=True)[0]
       
for key, value in Unique_dictGender.items():
    temp['Gender'] = np.where(temp['Gender'] == value, key, temp['Gender'])
    temp['Gender']=temp['Gender'].factorize(sort=True)[0]

# predicting a new value
st.subheader("Prediction of xgboost classifier is ")
y_pred = xgb_cl.predict(temp)
st.write(y_pred)
st.subheader("Prediction of naive bayes classifier is ")
y_preds = gnb.predict(temp)
st.write(y_preds)
st.subheader("Prediction of knn classifier is ")
preds = knn.predict(temp)
st.write(preds)
