#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import nltk
import string
import re

from nltk import wordpunct_tokenize
from nltk.stem import SnowballStemmer

from collections import Counter


# In[38]:


NUM_CLUSTER = 6


# In[34]:


nltk.download('stopwords')
nltk_stopwords = nltk.corpus.stopwords.words('english')
nltk_stopwords.extend(["entrepreneurship", "research", "entrepreneuri",
                       "entrepreneur", "entrepreneurs", "entrepreneurship", "entrepreneurial", 
                       "articl", 'may', 'work', 'high', 'result', 'model', 'firm', 'literature',
                       'paper', 'develop', 'use', 'growth', 'suggest', 'theori', 'provides', 'provide',
                       'start', 'startup', 'up', 'busi', "process", 'new', 'ventur', 'approach', 'role', 'higher', 'analysis',
                       'differ', 'effect', 'factor', 'support', 'relationship', 'also',
                       'data', 'level', 'increas', 'show', 'base', 'import',
                       'find', 'studi', 'c', 'b', 'elsevi' ,'reserv', 'right',
                      'research', 'firm', 'studi', 'use', 'theori', 'model',
                      'process', 'differ', 'find', 'result', 'suggest', 'base',
                      'busi', 'ventur', 'develop', 'new', 'effect', 'team', 'relat',
                      'growth', 'found', 'factor', 'may', 'learn', 'also', 'activ',
                      'level', 'high', 'influenc', 'provid', 'import', 'signific',
                      'make', 'elsevi', 'two', 'one', 'sampl', 'analysi', 'corpor', 'compani',
                      'implic', 'proactiv', 'paper', 'role', 'implic', 'rate', 'data', 'show',
                      'higher', 'becom', 'evid', 'surviv', 'increas', 'right', 'success'])
ps = SnowballStemmer('english')


# In[18]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer()
paper_clusters = list()


# In[39]:


for i in range(NUM_CLUSTER):
    data = pd.read_excel('D:/Thesis/Output/data_20210702/0702_ti1_ab2tfidf_w2v_6/'+ str(i)+'.xlsx')
    # make a new list and make every article lower and put them in li_n
    ### Title+abstract ###
    li_n = list()
    for line_ti, line_ab in zip(data['title'], data['abstract']):
        #li_n.append(line_ti.lower() + " " + line_ab.lower())
        line = line_ab.lower()
        line = re.sub(r"entrepreneurial orientation", "eo", line)
        line = re.sub(r"venture capital", "vc", line)
        li_n.append(line)

    allword = list()
    for article in li_n:
        punct_token = wordpunct_tokenize(article)
        # remove stopwords
        punct_token = [word for word in punct_token if word not in nltk_stopwords]
        # remove string.punctuation
        punct_token = [word for word in punct_token if word not in string.punctuation]
        # remove word that is not alphabat or number
        punct_token = [word for word in punct_token if word.isalnum() == True]
        punct_token = [ps.stem(word) for word in punct_token]
        punct_token = [word for word in punct_token if word not in nltk_stopwords]
        punct_token = [word.replace('economi', 'econom') for word in punct_token]
        punct_token = [word.replace('financi', 'financ') for word in punct_token]
        punct_token = [word.replace('strategi', 'strateg') for word in punct_token]

        allword.append(punct_token)

    li_word = list()
    for article in allword:
        for word in article:
            li_word.append(word)

    total_word_count = pd.DataFrame([Counter(li_word)]).T.sort_values(by=0, ascending=False).head(20)
    #test = pd.DataFrame(tfidf_matrix.toarray()).T.sort_values(by=0, ascending=False).head(20)
    #test.columns = ['value']       
    total_word_count.to_excel('D:/Thesis/Output/data_20210702/0702_ti1_ab2tfidf_w2v_6/keyword/'+ str(i) +'.xlsx')


# In[22]:


print(nltk_stopwords)


# In[ ]:




