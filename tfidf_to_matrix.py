#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[4]:


# mat file
#data = pd.read_excel('D:/Thesis/Output/0630_ti1_ab2tfidf_w2v_8/keyword/keyword_tfidf.xlsx')
data = pd.read_excel('D:/Thesis/Output/data_20210702/0704_ti1_ab2tfidf_w2v_6_remove_keyword/keyword_tfidf.xlsx')
test = data.columns[1:]
count = len(data.index)

#f = open('D:/Thesis/Output/0630_ti1_ab2tfidf_w2v_8/keyword/keyword_tfidf.mat', 'w')
f = open('D:/Thesis/Output/data_20210702/0704_ti1_ab2tfidf_w2v_6_remove_keyword/keyword_tfidf.mat', 'w')

f.write("*vertices "+ str(count + len(test)) + " " + " " + str(count) + "\n")

for i in range(count):
    f.write(" " + str(i + 1) + " \""+ str(i) + "\"\n")

count = count + 1

for i in test:
    f.write(" " + str(count) + " \""+ i + "\"\n")
    count = count + 1
    #print(" "+ str(i)+" \""+str(i)+"\"")

f.write("*matrix\n")
matrix_value = np.delete(data.values.tolist(), 0, axis=1)

for line in matrix_value:
    for i in line:
        value = 0 if i < 0 else i
        f.write(" " + str(value))
    f.write("\n")

f.close()

print(data.values)


# In[ ]:




