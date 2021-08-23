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


# In[48]:


NUM_CLUSTER = 7


# In[24]:


nltk.download('stopwords')
nltk_stopwords = nltk.corpus.stopwords.words('english')
nltk_stopwords.extend(['research', 'firm', 'studi', 'use', 'theori', 'model',
                      'process', 'differ', 'find', 'result', 'suggest', 'base',
                      'busi', 'ventur', 'develop', 'new', 'effect', 'team', 'relat',
                      'growth', 'found', 'factor', 'may', 'learn', 'also', 'activ',
                      'level', 'high', 'influence', 'provide', 'important', 'significant',
                      'make', 'elsevi', 'two', 'one', 'sample', 'analysis', 'corporate', 'company',
                      'implic', 'proactiv', 'paper', 'role', 'implic', 'rate', 'data', 'show',
                      'higher', 'become', 'evident', 'survive', 'increase', 'right', 'success',
                      'test', 'affect', 'support', 'value', 'like', 'understand', 'theoretical',
                      'impact', 'empirical', 'literature', 'approach', 'contribute', 'field',
                      'review', 'framework', 'article', 'emerging', 'change', 'inform', 'work',
                      'small', 'three', 'concept', 'conceptu', 'methodology', 'structure',
                      'perspect', 'discuss', 'within', 'offer', 'explor', 'design', 'identify',
                      'method', 'enterprise', 'present', 'issu', 'creation'])
ps = SnowballStemmer('english')


# In[4]:


def eucliDist(A, B):
    return np.sqrt(sum(np.power((A - B), 2)))


# In[34]:


def word2token(text):

    li_n = list()
    for line in text:
        #li_n.append(line_ti.lower() + " " + line_ab.lower())
        line = line.lower()
        line = re.sub(r"entre\w*", "", line)
        line = re.sub(r"new venture\w*", "", line)
        line = re.sub(r"startup\w*", "", line)
        line = re.sub(r"start\-up\w*", "", line)
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
        #punct_token = [ps.stem(word) for word in punct_token if word.isalnum() == True]
        #punct_token = [word for word in punct_token if word not in nltk_stopwords]
        #punct_token = [word.replace('economi', 'econom') for word in punct_token]

        allword.append(punct_token)
    return allword


# In[6]:


data = pd.read_excel('D:/Thesis/Raw data/20210702/20210702_10732_TI_AB_TC.xlsx')

ti_token = word2token(data['TI'])
ab_token = word2token(data['AB'])
tc = data['TC']
print(ti_token[15])


# In[35]:


from gensim.models import Word2Vec
#ab_allword = word2token()
model = Word2Vec(ti_token + ab_token, size=300, window=5, min_count=1, workers=4, sg=1)
#print(data['words'].head(5))


# In[43]:


from gensim.models import KeyedVectors
MODEL_PATH = "D:/Thesis/Model/"
model = KeyedVectors.load_word2vec_format(MODEL_PATH + 'GoogleNews-vectors-negative300.bin',
                                          binary=True)


# In[26]:


model = KeyedVectors.load_word2vec_format(MODEL_PATH + 'glove.6B\gensim_glove.6B.300d.txt', binary=False)


# In[15]:


import gensim.downloader
model = gensim.downloader.load('conceptnet-numberbatch-17-06-300')


# In[16]:


model = KeyedVectors.load_word2vec_format(MODEL_PATH + 'conceptnet/numberbatch-en.txt', binary=False)


# In[44]:


from UtilWordEmbedding import TfidfEmbeddingVectorizer

tfidf_vec_tr = TfidfEmbeddingVectorizer(model)


# In[45]:


tfidf_vec_tr.fit(ab_token)  # fit tfidf model first
ti_vec = tfidf_vec_tr.transform(ti_token)
ab_vec = tfidf_vec_tr.transform(ab_token)


# In[104]:


from UtilWordEmbedding import MeanEmbeddingVectorizer
mean_vec_tr = MeanEmbeddingVectorizer(model)
ti_vec = mean_vec_tr.transform(ti_token)
ab_vec = mean_vec_tr.transform(ab_token)
print(len(ti_vec))


# In[35]:


#ab_vec = mean_vec_tr.transform(ab_token)
tfidf_vec_tr.fit(ab_token)  # fit tfidf model first
ab_vec = tfidf_vec_tr.transform(ab_token)

#print(len(allword))
#print(len(tfidf_doc_vec))


# In[46]:


ti_ab_vec = (ab_vec * 2 + ti_vec) / 3

#ti_ab_vec = ab_vec


# In[49]:


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

print('doing clustering')

kmeans = KMeans(n_clusters=NUM_CLUSTER, init='k-means++', max_iter=200, n_init=20)
kmeans.fit(ti_ab_vec)
labels = kmeans.labels_


# In[50]:


#Center of weight calculation
cow_list = list()
for i in range(NUM_CLUSTER):
    fliter = (labels == i)
    vec = ti_ab_vec[fliter]
    weight = data[fliter].TC.values.tolist()

    vec_list = list()
    for i in range(len(ti_ab_vec[fliter])):
        vec_list.append(vec[i] * int(weight[i]))
        
    vec_pd = pd.DataFrame(vec_list)
    cow_list.append (vec_pd.sum()/data[fliter].TC.sum())
    
result_center_dist = []
result_centerofweight_dist = []

for i in range(len(ti_ab_vec)):
    result_center_dist.append(eucliDist(ti_ab_vec[i], kmeans.cluster_centers_[labels[i]]))
    result_centerofweight_dist.append(eucliDist(ti_ab_vec[i], cow_list[labels[i]]))


# In[51]:


# reduce the features to 2D
pca = PCA(n_components=3, random_state=0)
reduced_features = pca.fit_transform(ti_ab_vec)

# reduce the cluster centers to 2D
reduced_cluster_centers = pca.transform(kmeans.cluster_centers_)

# wiki_cl = pd.DataFrame(list(zip(data['TI'], data['AB'], labels, \
#                                 reduced_features[:, 0], \
#                                 reduced_features[:, 1], \
#                                 distance(reduced_features[:, 0], \
#                                          reduced_features[:, 1], \
#                                          reduced_cluster_centers[labels, 0], \
#                                          reduced_cluster_centers[labels, 1]))),
#                        columns=['title', 'abstract','cluster', 'x', 'y', 'distance'])
# df_output = pd.DataFrame(zip(data['TI'], data['AB']), columns=['TI', 'AB'])
# for k in range(10):
#    df_output[k+1] = KMeans(n_clusters=5, init='k-means++', max_iter=200, n_init=10).fit(test).labels_
# df_output.to_excel('C:/Users/TR814-Public/NLP/Model report/NLP_kmeans/Word2vec/20210421/trial_8/output_8.xlsx')

wiki_cl = pd.DataFrame(list(zip(data['TI'], data['AB'], data['TC'],                                 labels, result_center_dist, result_centerofweight_dist)),                        columns=['title', 'abstract', 'TC', 'cluster', 'distance', 'COW_dist'])

#distance(reduced_features[:, 0], reduced_features[:, 1], reduced_cluster_centers[labels, 0], reduced_cluster_centers[labels, 1])

#print(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1])

print("writing data")
wiki_cl.to_excel('D:/Thesis/Output/data_20210702/0704_ti1_ab2tfidf_w2v_7_remove_keyword.xlsx')


# In[42]:


from scipy.spatial.distance import cdist

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

SSE = []
for k in range(2, 15):
    estimator = KMeans(n_clusters=k, init='k-means++', max_iter=200, n_init=20)
    estimator.fit(ti_ab_vec)
    #SSE.append(estimator.inertia_ti_ab_vec
    SSE.append(sum(np.min(cdist(ti_ab_vec, estimator.cluster_centers_, 'euclidean'), axis=1)) / ti_ab_vec.shape[0])
X = range(2, 15)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X, SSE, 'o-')
plt.show()


# In[53]:


print(ps.stem('ee'))


# In[21]:


text = "entrepreneur, entrepreneurs, entrepreneurship, entrepreneurial"
text = re.sub(r"entre\w*", "", text)

print(text)


# In[ ]:




