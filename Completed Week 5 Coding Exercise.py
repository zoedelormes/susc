#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Week 5 


# In[46]:


def read_text(filename):

    with open(filename, 'r') as data:
        list_str = data.readlines()
        
    return(list_str)

#function worked on by Jason and Tiehan

read_text('nyc_temp_2010.txt')


# In[59]:


observed = [] #creating an empty list to place observed temps
for line in read_text('nyc_temp_2010.txt'):
    observed.append(line[0:2]) #removing the '\n' from the data and placing in list
#code contributed by Emmy


# In[60]:


observed #checking list 


# In[69]:


def print_data (temps):
    print("{:^5}  {:^16} ".format("Day", "Temperature (°F)")) #formatting header row
    for t in range(1,366):
        print("{:^5}  {:^16} ".format(t,observed[t]))


# In[70]:


print_data(observed)

