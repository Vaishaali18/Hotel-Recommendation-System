import pandas as pd
import numpy as np
import os
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import random
#import plotly.graph_objs as go
#import plotly.plotly as py
import plotly

import plotly.graph_objs as go
from plotly.offline import *

pd.options.display.max_columns = 30
from IPython.core.interactiveshell import InteractiveShell
#import plotly.figure_factory as ff
InteractiveShell.ast_node_interactivity = 'all'
from plotly.offline import iplot

os.chdir('C://Users//vaishaali//Documents//18C118//python/booking_com-travel_sample')
df=pd.read_csv('final_dataset.csv')

df.head()
print('We have ', len(df), 'hotels in the data')

def print_description(index):
    example = df[df.index == index][['hotel_description', 'property_name']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Name:', example[1])


def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(df['hotel_description'], 20)
df1 = pd.DataFrame(common_words, columns = ['hotel_description' , 'count'])
df1=df1.groupby('hotel_description').sum()['count'].sort_values()
df1=df1.to_frame()
#df1.iplot(kind='barh', yTitle='Count', linecolor='black', title='Top 20 words in hotel description before removing stop words')
#plt.barh(df1['hotel_description'], df1['count'], color ='red')
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(df['hotel_description'], 20)
df2 = pd.DataFrame(common_words, columns = ['hotel_description' , 'count'])
#df2.groupby('hotel_description').sum()['count'].sort_values().iplot(kind='barh', yTitle='Count', linecolor='black', title='Top 20 words in hotel description after removing stop words')
#plt.barh(df2['hotel_description'], df2['count'], color ='red')
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(df['hotel_description'], 20)
df3 = pd.DataFrame(common_words, columns = ['hotel_description' , 'count'])
#plt.barh(df3['hotel_description'], df3['count'], color ='red')
#df3.groupby('hotel_description').sum()['count'].sort_values(ascending=False).iplot(kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in hotel description before removing stop words')

def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(df['hotel_description'], 20)
df4 = pd.DataFrame(common_words, columns = ['hotel_description' , 'count'])
#plt.barh(df4['hotel_description'], df4['count'], color ='red')
#df4.groupby('hotel_description').sum()['count'].sort_values(ascending=False).iplot(kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in hotel description After removing stop words')


def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_trigram(df['hotel_description'], 20)
df5 = pd.DataFrame(common_words, columns = ['hotel_description' , 'count'])
#plt.barh(df5['hotel_description'], df5['count'], color ='red')
#df5.groupby('hotel_description').sum()['count'].sort_values(ascending=False).iplot(kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams in hotel description before removing stop words')

def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_trigram(df['hotel_description'], 20)
df6 = pd.DataFrame(common_words, columns = ['hotel_description' , 'count'])
#plt.barh(df6['hotel_description'], df6['count'], color ='red')
#df6.groupby('hotel_description').sum()['count'].sort_values(ascending=False).iplot(kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams in hotel description after removing stop words')

df['word_count'] = df['hotel_description'].apply(lambda x: len(str(x).split()))
desc_lengths = list(df['word_count'])
print("Number of descriptions:",len(desc_lengths),
      "\nAverage word count", np.average(desc_lengths),
      "\nMinimum word count", min(desc_lengths),
      "\nMaximum word count", max(desc_lengths))
import nltk
nltk.download('stopwords')

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text
    
df['desc_clean'] = df['hotel_description'].apply(clean_text)

df.set_index('property_name', inplace = True)
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(df['desc_clean'])
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index)

def recommendations(name, cosine_similarities = cosine_similarities):
    
    recommended_hotels = []
    
    # gettin the index of the hotel that matches the name
    idx = indices[indices == name].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar hotels except itself
    top_10_indexes = list(score_series.iloc[1:20].index)
    
    # populating the list with the names of the top 10 matching hotels
    for i in top_10_indexes:
        recommended_hotels.append(list(df.index)[i])
    res = []
    [res.append(x) for x in recommended_hotels if x not in res] 
    res.remove(name)
    return res

recommended_hotels=recommendations('Hotel Lighthouse')
print(recommended_hotels)




