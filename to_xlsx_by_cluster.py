#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np

NUM_CLUSTER = 7


# In[10]:


data = pd.read_excel('D:/Thesis/Output/data_20210702/0704_ti1_ab2tfidf_w2v_7_remove_keyword/0704_ti1_ab2tfidf_w2v_7_remove_keyword.xlsx')
for i in range(NUM_CLUSTER):
    fliter = (data["cluster"] == i)
    xlsx_pd = pd.DataFrame(data[fliter].values.tolist(),                            columns=['No', 'title', 'abstract', 'TC', 'cluster', 'distance',  'COW_dist'])
    xlsx_pd.to_excel('D:/Thesis/Output/data_20210702/0704_ti1_ab2tfidf_w2v_7_remove_keyword/'+ str(i) +'.xlsx', index=False)


# In[ ]:





# In[ ]:




