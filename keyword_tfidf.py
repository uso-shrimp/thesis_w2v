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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


# In[26]:


NUM_CLUSTER = 8


# In[8]:


nltk.download('stopwords')
nltk_stopwords = nltk.corpus.stopwords.words('english')
nltk_stopwords.extend(["entrepreneurship", "research", "entrepreneuri",
                       "entrepreneur", "entrepreneurs", "entrepreneurship", "entrepreneurial", 
                       "articl", 'may', 'work', 'high', 'result', 'model', 'firm', 'literature',
                       'paper', 'develop', 'use', 'growth', 'suggest', 'theori', 'provides', 'provide',
                       'start', 'startup', 'up', 'busi', "process", 'new', 'ventur', 'approach', 'role', 'higher', 'analysis',
                       'differ', 'effect', 'factor', 'support', 'relationship', 'also',
                       'data', 'level', 'increas', 'show', 'base', 'import',
                       'find', 'studi', 'c', 'b', 'elsevi' ,'reserv', 'right'])
ps = SnowballStemmer('english')


# In[27]:


paper_clusters = list()
for i in range(NUM_CLUSTER):
    #data = pd.read_excel('C:/Users/TR814-Public/NLP/data/Clustering/Entrepreneurship data 4138(final)_label_ti_ab.xlsx')
    #data = pd.read_excel('C:/Users/TR814-Public/NLP/Model report/NLP_kmeans/Word2vec/20210514/Google/0.xlsx')
    data = pd.read_excel('D:/Thesis/Output/0630_ti1_ab2tfidf_w2v_8/'+ str(i) +'.xlsx')

    # make a new list and make every article lower and put them in li_n
    ### Title+abstract ###
    li_n = list()
    for line_ti, line_ab in zip(data['title'], data['abstract']):
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
        #punct_token = [word for word in punct_token if word.isalnum() == True]
        punct_token = [ps.stem(word) for word in punct_token if word.isalnum() == True]
        punct_token = [word for word in punct_token if word not in nltk_stopwords]
        punct_token = [word.replace('economi', 'econom') for word in punct_token]
        punct_token = [word.replace('financi', 'financ') for word in punct_token]
        punct_token = [word.replace('strategi', 'strateg') for word in punct_token]

        allword.append(" ".join(punct_token))

    paper_clusters.append("".join(allword))

print(len(paper_clusters))
#paper_clusters.clear()


# In[29]:


tfidf_vec = TfidfVectorizer()
#tfidf_matrix = tfidf_vec.fit_transform(allword)
tfidf_matrix = tfidf_vec.fit_transform(paper_clusters)
tfidf_array = StandardScaler().fit_transform(tfidf_matrix.toarray())

#paper = pd.DataFrame(paper_clusters)
#paper.to_excel('D:/Thesis/Output/0524_ab_w2v/keyword/paper.xlsx')

total_word_count = pd.DataFrame(tfidf_array, columns=tfidf_vec.get_feature_names())
print(total_word_count.head(5))
keyword = pd.read_excel('D:/Thesis/Output/0630_ti1_ab2tfidf_w2v_8/keyword/keyword_list.xlsx')
keyword_count = total_word_count[keyword['keywords']]
#total_word_count.to_excel('C:/Users/TR814-Public/NLP/Model report/NLP_kmeans/Word2vec/20210514/Google/WordCount/9.xlsx')
keyword_count.to_excel('D:/Thesis/Output/0630_ti1_ab2tfidf_w2v_8/keyword/keyword_tfidf.xlsx')


# In[49]:





# In[ ]:




