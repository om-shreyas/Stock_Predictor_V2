#!/usr/bin/env python
# coding: utf-8

# In[7]:


from Prediction import *


# In[8]:


from Time_Series_Analysis import *


# In[9]:


def new_data(stock):
    create_model(stock)
    create_new(stock)


# In[13]:


def existing_data(stock):
    predict_model(stock)
    new = latest_prediction(stock)
    return(new)


# In[14]: