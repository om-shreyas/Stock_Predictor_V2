#!/usr/bin/env python
# coding: utf-8

# In[56]:


import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler


# In[57]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanAbsolutePercentageError as MAPE
from tensorflow.keras.metrics import MeanAbsolutePercentageError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


# In[58]:


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


# In[111]:


def test_predictions(X_test,y_test,model,path):
    test_predictions = model.predict(X_test).flatten()
    #test_results = pd.DataFrame(data={'test Predictions':test_predictions, 'Actuals':y_test})
    error = mean_absolute_percentage_error(y_test, test_predictions)
    
    fig, axe = plt.subplots(figsize=(7, 3.5), dpi=800)
    axe.plot(test_predictions)
    axe.plot(y_test)
    axe.legend(['Test','Actual'])
    axe.text(0,1.05,error,transform=axe.transAxes)
    
    fig.savefig(path+'/test_figure.png')
    plt.close()


# In[112]:


def df_to_X_y(df, window_size=5):
  df_as_np = df
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [[a] for a in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size]
    y.append(label)
  return np.array(X), np.array(y)


# In[113]:


def df_to_X_y_2D(df, window_size=5):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [r for r in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size][0]
    y.append(label)
  return np.array(X), np.array(y)


# In[114]:


def create_model_conv(X,y,window = 5,path = ''):
    X_train, y_train = X[:int(len(X)*0.7)],y[:int(len(X)*0.7)]
    X_val, y_val = X[int(len(X)*0.7):int(len(X)*0.85)],y[int(len(X)*0.7):int(len(X)*0.85)]
    X_test, y_test = X[int(len(X)*0.85):],y[int(len(X)*0.85):]
    
    model = Sequential()
    model.add(InputLayer((window, 1)))
    model.add(Conv1D(4096,2))
    model.add(Flatten())
    model.add(Dense(64, 'relu'))
    model.add(Dense(8, 'relu'))
    model.add(Dense(1, 'linear'))

    cp1 = ModelCheckpoint(path, save_best_only=True)
    model.compile(loss=MAPE(), optimizer=Adam(learning_rate=0.001), metrics=[MeanAbsolutePercentageError()])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15,callbacks=[cp1])

    model = load_model(path+'/')

    return(model)


# In[115]:


def create_model_LSTM_2D(X,y,window = 5,path = ''):
    X_train, y_train = X[:int(len(X)*0.7)],y[:int(len(X)*0.7)]
    X_val, y_val = X[int(len(X)*0.7):int(len(X)*0.85)],y[int(len(X)*0.7):int(len(X)*0.85)]
    X_test, y_test = X[int(len(X)*0.85):],y[int(len(X)*0.85):]

    scaler = StandardScaler()
    X_train = np.array(scaler.fit_transform(X_train))
    X_test = np.array(scaler.transform(X_test))
    X_val = np.array(scaler.transform(X_val))
    
    print(X_train.shape,y_train.shape)
    
    model = Sequential()
    model.add(InputLayer((window, X.shape[2])))
    model.add(Conv1D(4096,2))
    model.add(Flatten())
    model.add(Dense(64, 'relu'))
    model.add(Dense(8, 'relu'))
    model.add(Dense(1, 'linear'))

    cp1 = ModelCheckpoint('model/'+path, save_best_only=True)
    model.compile(loss=MAPE(), optimizer=Adam(learning_rate=0.0001), metrics=[MeanAbsolutePercentageError()])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, callbacks=[cp1])

    model = load_model(path+'/')
    
    plt.plot(model.predict(X_train).flatten()[-100:])
    plt.plot(y_train[-100:])
    plt.legend(['train','Actual'])

    return(model)


# In[116]:


def create_model(stock):
    date,X,y = genreate_stock_data(stock,'max')
    X_open,y_open = df_to_X_y(X['Open'],window_size=10)
    X_close,y_close = df_to_X_y(X['Close'],window_size=10)
    
    path_create ='models/'+stock+'/time_series'
    model_open  = create_model_conv(X_open,y_open,path=path_create+'/open',window=10)
    model_close  = create_model_conv(X_open,y_open,path=path_create+'/close',window=10)
    
    test_predictions(X_open[int(len(X_open)*0.85):],y_open[int(len(y_open)*0.85):],model_open,path_create)


# In[127]:


def predict_model(stock):
    path_exist ='models/'+stock+'/time_series'
    
    model_open = load_model(path_exist+'/open')
    model_close = load_model(path_exist+'/close')
    
    date,X,y = genreate_stock_data(stock,'1mo')
    
    closed = X['Close'][-10:]
    opened = X['Open'][-10:]
    
    for i in range(5):
        y_open = model_open.predict([np.array([opened[-10:]])])
        opened = np.append(opened,y_open.flatten()[0])
        y_close = model_close.predict([np.array([closed[-10:]])])
        closed = np.append(closed,y_close.flatten()[0])
        
    fig, axe = plt.subplots(figsize=(7, 3.5), dpi=800)
    axe.plot(closed)
    axe.plot(opened,linestyle=':')
    axe.legend(['Close','Open'])
    fig.savefig(path_exist+'/projected.png')


# In[ ]:




