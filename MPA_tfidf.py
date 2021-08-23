#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
import string
import re

from nltk import wordpunct_tokenize
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


# In[9]:


nltk.download('stopwords')
nltk_stopwords = nltk.corpus.stopwords.words('english')
nltk_stopwords.extend(['research', 'firm', 'studi', 'use', 'theori', 'model',
                      'process', 'differ', 'find', 'result', 'suggest', 'base',
                      'busi', 'ventur', 'develop', 'new', 'effect', 'team', 'relat',
                      'growth', 'found', 'factor', 'may', 'learn', 'also', 'activ',
                      'level', 'high', 'influenc', 'provid', 'import', 'signific',
                      'make', 'elsevi', 'two', 'one', 'sampl', 'analysi', 'corpor', 'compani',
                      'implic', 'proactiv', 'paper', 'role', 'implic', 'rate', 'data', 'show',
                      'higher', 'becom', 'evid', 'surviv', 'increas', 'right', 'success',
                      'test', 'affect', 'support', 'valu', 'like', 'understand', 'theoret',
                      'impact', 'empir', 'literatur', 'approach', 'contribut', 'field',
                      'review', 'framework', 'articl', 'emerg', 'chang', 'inform', 'work',
                      'small', 'three', 'concept', 'conceptu', 'methodolog', 'structur',
                      'perspect', 'discuss', 'within', 'offer', 'explor', 'design', 'identifi',
                      'method', 'enterpris', 'present', 'issu', 'creation', 'dimens', 'formal',
                      'howev', 'take', 'variabl', 'among', 'need'])
ps = SnowballStemmer('english')


# In[10]:


paper_clusters = list()
for i in range(6):
    #data = pd.read_excel('C:/Users/TR814-Public/NLP/data/Clustering/Entrepreneurship data 4138(final)_label_ti_ab.xlsx')
    #data = pd.read_excel('C:/Users/TR814-Public/NLP/Model report/NLP_kmeans/Word2vec/20210514/Google/0.xlsx')
    data = pd.read_excel('D:/Thesis/Output/data_20210702/0704_ti1_ab2tfidf_w2v_6_remove_keyword/'+ str(i) +'.xlsx')

    # make a new list and make every article lower and put them in li_n
    ### Title+abstract ###
    li_n = list()
    for line_ti, line_ab in zip(data['title'], data['abstract']):
        line = line_ab.lower()
        line = re.sub(r"entre\w*", "", line)
        line = re.sub(r"new venture\w*", "", line)
        line = re.sub(r"startup\w*", "", line)
        line = re.sub(r"start\-up\w*", "", line)
        line = re.sub(r"entrepreneurial orientation", "eo", line)
        line = re.sub(r"corporate entrepreneurship", "ce", line)
        line = re.sub(r"venture capital", "vc", line)
        line = re.sub(r"vcs", "vc", line)
         
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
        punct_token = [word.replace('behaviour', 'behavior') for word in punct_token]

        allword.append(" ".join(punct_token))

    paper_clusters.append("".join(allword))

print(len(paper_clusters))
#paper_clusters.clear()


# In[11]:


tfidf_vec = TfidfVectorizer()
#tfidf_matrix = tfidf_vec.fit_transform(allword)
tfidf_matrix = tfidf_vec.fit_transform(paper_clusters)

word_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vec.get_feature_names()).T
#print(word_tfidf.head(5))
#keyword_list = np.array()


# In[12]:


keyword_list = list()
for i in range(6):
    #print(word_tfidf[i])
    tmp = word_tfidf[i].sort_values(ascending=False).head(25)
    keyword_list = np.append(keyword_list, np.array(tmp.index.tolist()))
    tmp.to_excel("D:/Thesis/Output/data_20210702/0704_ti1_ab2tfidf_w2v_6_remove_keyword/tfidf/"+ str(i) +'.xlsx')

keyword_list = np.unique(keyword_list).tolist()
print(keyword_list)


# In[13]:


keyword_tfidf = word_tfidf.T[keyword_list]
tfidf_array = StandardScaler().fit_transform(keyword_tfidf.values)
keyword_tfidf = pd.DataFrame(tfidf_array, columns=keyword_tfidf.columns)
keyword_tfidf.to_excel('D:/Thesis/Output/data_20210702/0704_ti1_ab2tfidf_w2v_6_remove_keyword/keyword_tfidf.xlsx')


# In[ ]:




