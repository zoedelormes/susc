#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Week 5 


# In[32]:


def read_text(filename):

    with open(filename, 'r') as data:
        list_str = data.readlines()
        
    return(list_str)

#function worked on by Jason and Tiehan

read_text('nyc_temp_2010.txt')


# In[33]:


observed = [] #creating an empty list to place observed temps
for line in read_text('nyc_temp_2010.txt'):
    observed.append(line[0:2]) #removing the '\n' from the data and placing in list
#code contributed by Emmy


# In[34]:


observed #checking list 


# In[35]:


def print_data (temps):
    print("{:^5}  {:^16} ".format("Day", "Temperature (°F)")) #formatting header row
    for t in range(1,366):
        print("{:^5}  {:^16} ".format(t,observed[t-1])) #print the observed temps on day of the year


# In[36]:


print_data(observed)


# In[ ]:





# In[ ]:


# Week 6 Coding Exercise


# In[1]:


import numpy as np


# In[2]:


y0 = 55 #assigned values
A = 25
t0 =200
T = 365


# In[119]:


def read_text(filename):
    temp_data = []
    with open(filename, 'r') as data:
        observed_t = data.readlines() #returns each line in file in a list
    return observed_t


# In[10]:


def lsq_fit(filename):

  with open(filename, 'r') as data:
    observation_t = data.readlines() #returns each line in file in a list
    int_obs = [int(i) for i in observation_t]

  model_value = []
  residual_value = []

  day = list(range(1,366))

  for t in day:
    y_t = y0 + A * np.cos(2 * np.pi * (t - t0) / T)
    model_value.append(round(float(y_t),1)) #adding y_t to initialized list, making it into floating value and rounding
    residual_t = float(y_t) - int_obs[t-1]
    residual_value.append(round(float(residual_t),1))

  return model_value, residual_value


# In[11]:


print("{:^5} | {:^16} | {:^5} | {:^8}".format("Day", "Temperature (°F)", "y_t", "Residuals")) #formatting header row
print("-"*45) #creates line between header and data


# In[12]:


lsq_fit('nyc_temp_2010.txt')


# In[163]:


#code developed by Lexi and Jason, debugged by Emmy
def print_data (day,observed,model):
    print("{:^5} | {:^16} | {:^6} | {:^8}".format("Day", "Temperature (°F)", "y_t (°F)", "Residuals")) #formatting header row
    print("-"*48) #creates line between header and data
    for t in range(1,366): #using t to index 
        print("{:^5} | {:^16.1f} | {:^8.1f} | {:>6.1f}".format(t,observed_temp[t-1],model_value[t-1],residual_value[t-1])) #print the observed temps on day of the year


# In[164]:


filename = 'nyc_temp_2010.txt'
observed_temp = read_text(filename) #temps are retrieved as a string
for i in range (0,365):
    observed_temp[i] = float(observed_temp[i][0:2]) #converting observed values to float numbers
model_value, residual_value = lsq_fit(filename)


# In[165]:


print_data(observed_temp, model_value, residual_value)


# In[ ]:





# In[ ]:


# Week 7 Coding Exercise 


# In[99]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#importing necessary packages


# In[281]:


#code worked on by Emmy + Lexi
def plot_ts(temp, y_t, residuals):
    day = list(range(1,366))
    fig,(ax1,ax2) = plt.subplots(2) #creates two stacked subplots
    ax1.plot(day,temp,label=('observed'))
    ax1.plot(day,y_t,label=('model')) #labeling the different plots
    ax1.set_title('Observed vs Model Temperatures') #naming the subplot
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Temperature (°F)')
    ax2.plot(day,residuals,'tab:red') #plotting residuals in red
    ax2.set_title('Pre-fit Model Residuals') #naming the subplot
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Temperature (°F)')
    plt.subplots_adjust(hspace=0.6) #increasing the sapce between subplots
    ax1.legend(loc='upper left') #adjusts location of labels
    fig.savefig('Figure1.pdf') #saves the figure to directory


# In[282]:


plot_ts(observed_temp,model_value,residual_value)


# In[ ]:




