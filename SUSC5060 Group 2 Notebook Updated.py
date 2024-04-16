#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Group 2: Jason, Emmy, Lexi, and Tiehan


# In[1]:


#Week 5 Coding Exercise


# In[5]:


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





# In[ ]:


# Week 9 Coding Exercise


# In[1]:


# putting in partial code, it opens and reads.

import netCDF4 as nc
import numpy as np

def read_netcdf(file_name):
    # Open the NetCDF file with read only
    root = nc.Dataset(file_name, 'r')


    return latitudes, longitudes

### Get the variables as a Python dictionary
    variables = root.variables


    # Get the latitude and longitude
    latitudes = root.variables['latitude'][:]
    longitudes = root.variables['longitude'][:]


    # Trying to print the latitiude and longitudes retrieved (not working)
    print (latitude)
    print (longitude)

# Define file name
file_name = 'ERA5_SAT_195901-202112.nc'


# In[2]:


import xarray as xr 
#xarray is used because netCDF4 is not working on my computer
import numpy as np

def read_netcdf(file_name):
    # Open the NetCDF file with read only
    root = xr.open_dataset(file_name)
    #return latitudes, longitudes

### Get the variables as a Python dictionary
    #variables = root.variables

    # Get the latitude and longitude
    latitudes = root.variables['latitude'][:]
    longitudes = root.variables['longitude'][:]

    # Trying to print the latitiude and longitudes retrieved (not working)
    print (latitude)
    print (longitude)

# Define file name
file_name = 'ERA5_SAT_195901-202112.nc'


# In[3]:


import xarray as xr 
#xarray is used because netCDF4 is not working on my computer
import numpy as np

def read_netcdf(file_name):
    with xr.open_dataset("file_name") as root:
        print(root.variables['latitude'][:])
        
file_name = 'ERA5_SAT_195901-202112.nc'


# In[4]:


read_netcdf(file_name)


# In[6]:


print('ERA5_SAT_195901-202112.nc'[:4])


# In[ ]:





# In[ ]:


# Week 10 Coding Exercise


# In[6]:


import netCDF4 as nc


# In[2]:


dataset = nc.Dataset('ERA5_SAT_195901-202112.nc')

#convert temp
def convert_kelvin_to_fahrenheit(kelvin_temp):
    return (kelvin_temp - 273.15) * 1.8 + 32


# In[ ]:


nyc_sat = ds['t2m'].sel(latitude=40.7128, longitude=360-74.0060, method='nearest')\

nyc_sat_fah = convert_kelvin_to_fahrenheit(nyc_sat)


# In[ ]:


annual_mean_sat = nyc_sat_fahrenheit.resample(time='A').mean()

def plot_hist(annual_mean_sat):
    # Convert the xarray DataArray to a numpy array
    data = annual_mean_sat.values

    plt.figure(figsize=(10, 6))
    # Determine the bin edges for a bin size of 0.5°F
    bin_edges = np.arange(np.nanmin(data), np.nanmax(data) + 0.5, 0.5)
    plt.hist(data, bins=bin_edges, density=True, alpha=0.6, color='g')


    # Label the axes and add a title
    plt.xlabel('Temperature (°F)')
    plt.ylabel('Probability Density')
    plt.title('Annual-Mean SAT Probability Density Function for NYC')


    # Save the figure
    plt.savefig('Figure2.pdf')
    plt.show()


# In[ ]:





# In[ ]:


# Week 12 Coding Exercise


# In[5]:


import pandas as pd
from scipy.optimize import least_squares

# Load data under here

data = pd.read_csv('nyc_temp_2010.txt')
time_points = data[''].values
temperatures = data[''].values

# Constants
T = 365  # period in days
x1_prior, x2_prior, x3_prior = 55, 25, 200

def lsq_fit(days, temperatures):
    A = np.column_stack([np.ones(len(days)), np.cos(2 * np.pi * days / 365), np.sin(2 * np.pi * days / 365)])

    x, _, _, _ = np.linalg.lstsq(A, temperatures, rcond=None)

    x1, x4, x5 = x

    post_fit_model = x1 + x4 * np.cos(2 * np.pi * days / 365) + x5 * np.sin(2 * np.pi * days / 365)

    post_fit_residuals = temperatures - post_fit_model
    
    return (x1, x4, x5), post_fit_model, post_fit_residuals


# In[ ]:




