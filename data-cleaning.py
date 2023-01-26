import pandas as pd
import numpy as np
import nltk as n
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
from sklearn.cluster import KMeans
from nltk.corpus import stopwords, wordnet

data = pd.read_csv('original_data.csv', encoding='cp1252')

data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6'], axis=1)

data[data.isna().any(axis=1)]
#Since it's one value and this user has multiple posts, we will remove it. 
data = data.dropna()

#making AUTHID easier to read
data['#AUTHID'] = data['#AUTHID'].rank(method='dense').astype(int)

# Combining by user
data_byuser = data.groupby(['#AUTHID'])['STATUS'].apply(lambda x: ' '.join(x)).reset_index()
counts = pd.value_counts(data['#AUTHID'].values.ravel())
df_counts = pd.DataFrame(counts) # wrap pd.Series to pd.DataFrame
df_counts = df_counts.reset_index()
df_counts.columns = ['#AUTHID', 'post_count']

data_byuser = pd.merge(data_byuser, df_counts)

# text cleaning STATUS variable
# remove the hashtags and unwanted characters.
def clean_text(df, text_field):
    df[text_field] = df[text_field].str.lower() #lower case everything
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r'[^a-zA-Z0-9 ]', ' ', elem)) 
    return df

data_byuser_clean = clean_text(data_byuser, 'STATUS')

# Create lists
data_status_rows = [l.split("\n")[0] for l in data['STATUS']] #for whole dataset in case we need this later
data_byuser_status_rows = [l.split("\n")[0] for l in data_byuser['STATUS']] #by user

word_list = []
for i in data_byuser_status_rows:
    word_list.append(n.word_tokenize(i)) #Combined by user aka AUTHID

data_byuser_clean['status_words'] = word_list

# going back to 'data' for a sec
big5 = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
for i in big5:
    data[i] = data[i].eq('y').mul(1)

cols = ['#AUTHID', 'NETWORKSIZE', 'BETWEENNESS',
       'NBETWEENNESS', 'DENSITY', 'BROKERAGE', 'NBROKERAGE', 'TRANSITIVITY',
       'cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']

data_250 = data[cols]
data_250 = data_250.drop_duplicates()
data_byuser = pd.merge(data_byuser_clean, data_250)


# Creating new variables
# character length per sentence 
data_byuser['total_words'] = data_byuser['status_words'].str.len()

# avg words per pst
data_byuser['avg_words'] = data_byuser['total_words']/data_byuser['post_count']

me_words = ['i','me','my','myself', 'mine','we','our','ours','ourselves']
me_total = []
for i in data_byuser['status_words']:
    c = len([k for k in i if k in me_words])
    me_total.append(c)
# total ME words
data_byuser['me_words'] = me_total

data_byuser['me_perpost'] = data_byuser['me_words']/data_byuser['post_count']

data_byuser.to_csv('data_byuser.csv')






