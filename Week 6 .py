#!/usr/bin/env python
# coding: utf-8

# In[37]:


import math


# In[38]:


y0 = 55
A = 25
t0 = 200
T = 365


# In[49]:


def model_value(t): 
    model= y0 + A* math.cos(2*math.pi)*(t-t0 /T)
    return model


# In[57]:


import pandas as pd
import numpy as np


# In[58]:


t = np.array([])


# In[61]:


for i in range(1,365):
    t = np.append(t,i)


# In[62]:


model_value(t)


# In[63]:





# In[ ]:




