#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import re


# In[3]:


from functools import partial
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="AIzaSyAZ3GSLFnUR3XmAd_JkUre3Tyi2K8M1GXg")


# In[5]:


geocode = partial(geolocator.geocode, language="es")
print(geocode("HOBOKEN", language="en"))


# In[56]:


data = pd.read_excel('D:/Thesis/Raw data/Edge betweenness clustering/data/2.xlsx')

country_list = list()
for line_add in data['PI']:
    #print(line_add)
    geo = geolocator.geocode(line_add)
    location = geolocator.reverse(str(geo.latitude)+","+str(geo.longitude), language="en")
    country = location.raw['address'].get('country')
    country_list.append(country)
    print(country)

print(country_list)


# In[54]:


print(len(country_list))


# In[55]:


tmp = pd.DataFrame(country_list)
tmp.to_excel("D:/Thesis/Raw data/Edge betweenness clustering/data/tmp.xlsx")


# In[ ]:




