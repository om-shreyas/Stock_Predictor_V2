#!/usr/bin/env python
# coding: utf-8

# In[131]:


import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error


# In[132]:


def genreate_stock_data(stock_name,period):
    company = yf.Ticker(stock_name)
    df = company.history(period=period)
    
    #Sepreate the dates of the stock for further use if required
    date = (df.reset_index())['Date']
    #-------------------------------------------------------------------------------------------------------------
    
    #Remove data object from the database
    df = (df.reset_index()).drop('Date',axis=1)
    #-------------------------------------------------------------------------------------------------------------
    
    #Create the current trend of the current stock
    t = []
    for i in range(len(df)):
        if(df['Open'].iloc[i]-df['Close'].iloc[i]>0):
            t.append("DOWN")
        else:
            t.append("UP")
            
    df['Trend']=t
    #-------------------------------------------------------------------------------------------------------------
    
    #Create the trend history of the stock
    th=[0]
    current_trend = df['Trend'].iloc[0]
    count = 0
    for i in range(1,len(df)):
        if(current_trend==df['Trend'].iloc[i]):
            count+=1
            th.append(count)
        else:
            count = 0
            th.append(count)
            current_trend = df['Trend'].iloc[i]
            
    df["Trend history"]=th
    #-------------------------------------------------------------------------------------------------------------
    
    #Create the comparison to the change since last peak
    p=[]
    v=[]
    if(df['Trend'].iloc[0]=="DOWN"):
        peak = df['Open'].iloc[0]
        valley = df['Close'].iloc[0]
    else:
        valley = df['Open'].iloc[0]
        peak = df['Close'].iloc[0]

    for i in range(0,len(df)):

        p.append(peak-df['Close'].iloc[i])
        v.append(valley-df['Close'].iloc[i])

        if peak<df['Close'].iloc[i]:
            peak = df['Close'].iloc[i]
        if valley>df['Close'].iloc[i]:
            valley = df['Close'].iloc[i]
            
    df['Change since last peak']=p
    df['Change since last drop']=v
    #-------------------------------------------------------------------------------------------------------------
    
    #Create the local change of daily stock
    c=[]
    cm=[]
    for i in range(0,len(df)):
        c.append(df['Open'].iloc[i]-df['Close'].iloc[i])
        cm.append(df['High'].iloc[i]-df['Low'].iloc[i])

    df['Local Change']=c
    df['Local range of stock price']=cm
    #-------------------------------------------------------------------------------------------------------------
    
    #Create dummy variable for the categorical variable
    df = pd.get_dummies(df,drop_first=True)
    #-------------------------------------------------------------------------------------------------------------
    
    #Create the label for the change from tommorow, the data that needs to be predicted
    next_day_change = df[1:]
    next_day_change = next_day_change['Local Change']
    #-------------------------------------------------------------------------------------------------------------
    
    #Remove The last column as the it is redundant
    df.drop(index=df.index[-1],axis=0,inplace=True)
    #-------------------------------------------------------------------------------------------------------------
    
    #Break the data into features and labels
    X = df
    y = next_day_change
    #-------------------------------------------------------------------------------------------------------------
    
    return(date,X,y)


# In[133]:


def genreate_stock_last_data(stock_name):
    company = yf.Ticker(stock_name)
    df = company.history(period='1mo')
    
    #Sepreate the dates of the stock for further use if required
    date = (df.reset_index())['Date']
    #-------------------------------------------------------------------------------------------------------------
    
    #Remove data object from the database
    df = (df.reset_index()).drop('Date',axis=1)
    #-------------------------------------------------------------------------------------------------------------
    
    #Create the current trend of the current stock
    t = []
    for i in range(len(df)):
        if(df['Open'].iloc[i]-df['Close'].iloc[i]>0):
            t.append("DOWN")
        else:
            t.append("UP")
            
    df['Trend']=t
    #-------------------------------------------------------------------------------------------------------------
    
    #Create the trend history of the stock
    th=[0]
    current_trend = df['Trend'].iloc[0]
    count = 0
    for i in range(1,len(df)):
        if(current_trend==df['Trend'].iloc[i]):
            count+=1
            th.append(count)
        else:
            count = 0
            th.append(count)
            current_trend = df['Trend'].iloc[i]
            
    df["Trend history"]=th
    #-------------------------------------------------------------------------------------------------------------
    
    #Create the comparison to the change since last peak
    p=[]
    v=[]
    if(df['Trend'].iloc[0]=="DOWN"):
        peak = df['Open'].iloc[0]
        valley = df['Close'].iloc[0]
    else:
        valley = df['Open'].iloc[0]
        peak = df['Close'].iloc[0]

    for i in range(0,len(df)):

        p.append(peak-df['Close'].iloc[i])
        v.append(valley-df['Close'].iloc[i])

        if peak<df['Close'].iloc[i]:
            peak = df['Close'].iloc[i]
        if valley>df['Close'].iloc[i]:
            valley = df['Close'].iloc[i]
            
    df['Change since last peak']=p
    df['Change since last drop']=v
    #-------------------------------------------------------------------------------------------------------------
    
    #Create the local change of daily stock
    c=[]
    cm=[]
    for i in range(0,len(df)):
        c.append(df['Open'].iloc[i]-df['Close'].iloc[i])
        cm.append(df['High'].iloc[i]-df['Low'].iloc[i])

    df['Local Change']=c
    df['Local range of stock price']=cm
    #-------------------------------------------------------------------------------------------------------------
    
    #Create dummy variable for the categorical variable
    df = pd.get_dummies(df,drop_first=True)
    #-------------------------------------------------------------------------------------------------------------
    
    #Remove The last column as the it is redundant
    latest_data = df.iloc[-1:]
    return(latest_data)


# In[134]:


def poly_convert(X,n):
    polynomial_converter = PolynomialFeatures(degree=n,include_bias=False)
    poly_features = polynomial_converter.fit_transform(X)
    return(poly_features)


# In[135]:


def evaluate_stock(stock,duration,iterations):
    
    date, X, y = genreate_stock_data(stock,duration)
    
    train_rmse_errors = []
    test_rmse_errors = []
    for i in range(1,iterations):
        polynomial_converter = PolynomialFeatures(degree=i,include_bias=False)
        poly_features = polynomial_converter.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)
        model = LinearRegression(fit_intercept=True)
        model.fit(X_train,y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_RMSE = mean_absolute_percentage_error(y_train,train_pred)
        test_RMSE = mean_absolute_percentage_error(y_test,test_pred)

        train_rmse_errors.append(train_RMSE)
        test_rmse_errors.append(test_RMSE)
        
    plt.plot(range(1,iterations),train_rmse_errors,label='TRAIN')    
    plt.plot(range(1,iterations),test_rmse_errors,label='TEST')
    plt.xlabel("Polynomial Complexity")
    plt.ylabel("RMSE")
    plt.legend()


# In[136]:


def create_model(stock,duration,n):
    date, X, y = genreate_stock_data(stock,duration)
    
    X = poly_convert(X,n)
    
    model = LinearRegression(fit_intercept=True)
    model.fit(X,y)
    
    return(model)


# In[137]:


def model_predict(model,n,X,y,path):
    X_poly = poly_convert(X,n)
    y_predict = model.predict(X_poly)
    train_error = mean_absolute_percentage_error(y,y_predict)
    
    fig, axes = plt.subplots(figsize=(7, 3.5), dpi=800)
    
    print(train_error)
    
    if not os.path.isdir(path):
        os.makedirs(path)

    axes.plot(X.index,y,y_predict)
    axes.legend(["Actual","Predicted"])
    axes.text(0,1.05,train_error,transform=axes.transAxes)
    fig.savefig(path+'model_analysis.png')


# In[138]:


def create_new(stock):
    model = create_model(stock,'max',3)
    date,X,y= genreate_stock_data(stock,'5mo')
    path = 'models/'+stock+'/regression/'
    model_predict(model,3,X,y,path)
    
    joblib.dump(model,path+'regression_model.pkl')


# In[139]:


def latest_prediction(stock):
    path = 'models/'+stock+'/regression/'
    model = joblib.load(path+'regression_model.pkl')
    last = genreate_stock_last_data(stock)
    X_poly = poly_convert(last,3)
    y = model.predict(X_poly)
    return(y[0])


# In[ ]:




