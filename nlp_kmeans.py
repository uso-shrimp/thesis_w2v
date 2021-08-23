#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
import re
import gc

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from tqdm import tqdm
from UtilWordEmbedding import TfidfEmbeddingVectorizer


# In[4]:


MODEL_PATH = "D:/Thesis/Model/"


# In[5]:


def text2vec(text):
    return np.mean([model[x] for x in text.split() if x in model.vocab], axis=0).reshape(1,-1)


# In[6]:


def text_to_wordlist(text, remove_stopwords=True, stem_words=True):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r",", " ", text)
    #text = re.sub(r"entrepreneur", " ", text)
    #text = re.sub(r"entrepreneurial", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r"\:", " : ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r" and ", " ", text)
    text = re.sub(r" and ", " ", text)
    text = re.sub(r" the ", " ", text)
    text = re.sub(r" this ", " ", text)
    text = re.sub(r" paper ", " ", text)
    text = re.sub(r" research ", " ", text)
    text = re.sub(r" study ", " ", text)
    #text = re.sub(r" firm ", " ", text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return (text)


# In[7]:


def eucliDist(A, B):
    return np.sqrt(sum(np.power((A - B), 2)))


# In[20]:


tqdm.pandas()

data = pd.read_excel('D:/Thesis/output.xlsx')

data['words'] = data.abstract.progress_apply(text_to_wordlist)
#data['words_ti'] = data.TI.progress_apply(text_to_wordlist)

#print(data.head(5).T)


# In[11]:


print("Loading model")
# Convert glove to word2vec
input_file = 'glove.6B\glove.6B.300d.txt'
output_file = 'glove.6B\gensim_glove.6B.300d.txt'
glove2word2vec(MODEL_PATH + input_file, MODEL_PATH + output_file)


# In[13]:


# Test Glove model
output_file = 'glove.6B\gensim_glove.6B.300d.txt'
#model = KeyedVectors.load_word2vec_format(MODEL_PATH + output_file, binary=False)
model = KeyedVectors.load_word2vec_format(MODEL_PATH + 'GoogleNews-vectors-negative300.bin',
                                          binary=True)


# In[14]:


print("Converting to vector")
data['vectors'] = data.words.progress_apply(text2vec)
#print(data['vectors'])
#data['vectors_ti'] = data.words_ti.progress_apply(text2vec)
#print(data['vectors_ti'])
#test = np.array(data['vectors'].values) / 2.0 \
#       + np.array(data['vectors_ti'].values) / 2.0
test = np.array(data['vectors'].values)
#       + np.array(data['vectors_ti'].values) / 2.0
test = np.concatenate(test)
test = StandardScaler().fit_transform(test)
#print(test[0])
#concat = pd.concat((data['vectors'].values, data['vectors_ti'].values), axis=1)
#concat = np.concatenate((data['vectors'].values, data['vectors_ti'].values), axis=1)
#print(concat)
#data['all'] = concat.mean(axis=1)
#data['all'] = np.mean(concat, axis=0)
#print(data['all'])

print('doing clustering')
#X_train = np.concatenate(data['vectors'].values)
#print(X_train[0])
# print(data['vectors'].head(5).values)
# print(data['vectors'].size)
# print(X_train.size)
# print(len(X_train))
# print(np.size(X_train, 0))
# print(np.size(X_train, 1))

kmeans = KMeans(n_clusters=10, init='k-means++', max_iter=200, n_init=20)
kmeans.fit(test)
labels = kmeans.labels_

result = []

for i in range(len(test)):
    result.append(eucliDist(test[i], kmeans.cluster_centers_[labels[i]]))

#print(len(test))
#print(len(result))

# reduce the features to 2D
pca = PCA(n_components=3, random_state=0)
reduced_features = pca.fit_transform(test)

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

wiki_cl = pd.DataFrame(list(zip(data['title'], data['abstract'], labels, result)),                         columns=['title', 'abstract', 'cluster', 'distance'])

#distance(reduced_features[:, 0], reduced_features[:, 1], reduced_cluster_centers[labels, 0], reduced_cluster_centers[labels, 1])

#print(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1])

print("writing data")
wiki_cl.to_excel('D:/Thesis/Output/0524_ab_w2v.xlsx')


# plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=kmeans.predict(test))
# cluster_0 = plt.scatter(reduced_cluster_centers[0, 0], reduced_cluster_centers[0, 1], marker='x', s=100, c='b')
# cluster_1 = plt.scatter(reduced_cluster_centers[1, 0], reduced_cluster_centers[1, 1], marker='x', s=100, c='m')
# cluster_2 = plt.scatter(reduced_cluster_centers[2, 0], reduced_cluster_centers[2, 1], marker='x', s=100, c='r')
# cluster_3 = plt.scatter(reduced_cluster_centers[3, 0], reduced_cluster_centers[3, 1], marker='o', s=100, c='b')
# cluster_4 = plt.scatter(reduced_cluster_centers[4, 0], reduced_cluster_centers[4, 1], marker='o', s=100, c='r')
#
# plt.legend((cluster_0, cluster_1, cluster_2, cluster_3, cluster_4),
#            ('0', '1', '2', '3', '4'),
#            scatterpoints=1,
#            loc='upper right',
#            ncol=3,
#            fontsize=8)
# plt.show()

# ##### Silhouette score #####
# scores = []
# for k in range(2, 20):
#     labels = KMeans(n_clusters=k, init='k-means++', max_iter=200, n_init=10).fit(test).labels_
#     score = metrics.silhouette_score(test, labels)
#     scores.append(score)
#
# plt.plot(list(range(2, 20)), scores)
# plt.xticks(range(0, 22, 1))
# plt.grid(linestyle='--')
# plt.xlabel("Number of Clusters Initialized")
# plt.ylabel("Silhouette Score")
# plt.show()

##### Sum of the squared errors #####
SSE = []
for k in range(2, 10):
    estimator = KMeans(n_clusters=k, init='k-means++', max_iter=200, n_init=20)
    estimator.fit(test)
    #SSE.append(estimator.inertia_)
    SSE.append(sum(np.min(cdist(test, estimator.cluster_centers_, 'euclidean'), axis=1)) / test.shape[0])
X = range(2, 10)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X, SSE, 'o-')
plt.show()


# In[24]:


from UtilWordEmbedding import TfidfEmbeddingVectorizer

tfidf_vec_tr = TfidfEmbeddingVectorizer(model)
tfidf_vec_tr.fit(data['words'])  # fit tfidf model first
tfidf_doc_vec = tfidf_vec_tr.transform(data['words'])

print(len(tfidf_doc_vec))


# In[ ]:




