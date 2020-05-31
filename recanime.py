# -*- coding: utf-8 -*-
"""
Recanime 1.0
Updates in the future
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.show()

anime = pd.read_csv('animes.csv')

#preprocessing
anime = anime.drop(['img_url','link'],axis=1)
anime = anime.drop_duplicates(subset='title')

anime['Action']=[0]*16214
anime['Adventure']=[0]*16214
anime['Cars']=[0]*16214
anime['Comedy']=[0]*16214
anime['Dementia']=[0]*16214
anime['Demons']=[0]*16214
anime['Drama']=[0]*16214
anime['Ecchi']=[0]*16214
anime['Fantasy']=[0]*16214
anime['Game']=[0]*16214
anime['Harem']=[0]*16214
anime['Hentai']=[0]*16214
anime['Historical']=[0]*16214
anime['Horror']=[0]*16214
anime['Josei']=[0]*16214
anime['Kids']=[0]*16214
anime['Magic']=[0]*16214
anime['Martial Arts']=[0]*16214
anime['Mecha']=[0]*16214
anime['Military']=[0]*16214
anime['Music']=[0]*16214
anime['Mystery']=[0]*16214
anime['Parody']=[0]*16214
anime['Police']=[0]*16214
anime['Psychological']=[0]*16214
anime['Romance']=[0]*16214
anime['Samurai']=[0]*16214
anime['School']=[0]*16214
anime['Sci-Fi']=[0]*16214
anime['Seinen']=[0]*16214
anime['Shoujo']=[0]*16214
anime['Shoujo Ai']=[0]*16214
anime['Shounen']=[0]*16214
anime['Shounen Ai']=[0]*16214
anime['Slice of Life']=[0]*16214
anime['Space']=[0]*16214
anime['Sports']=[0]*16214
anime['Super Power']=[0]*16214
anime['Supernatural']=[0]*16214
anime['Thriller']=[0]*16214
anime['Vampire']=[0]*16214
anime['Yaoi']=[0]*16214
anime['Yuri']=[0]*16214

anime = anime.reset_index()
anime = anime.drop('index',axis=1)

for i in range(0,len(anime)):
    genre = anime.iloc[i]['genre']
    
    if 'Action' in genre:
        anime.at[i,'Action']=1
        
    if 'Adventure' in genre:
        anime.at[i,'Adventure']=1
    
    if 'Cars' in genre:
        anime.at[i,'Cars']=1
        
    if 'Comedy' in genre:
        anime.at[i,'Comedy']=1
    
    if 'Dementia' in genre:
        anime.at[i,'Dementia']=1
    
    if 'Demons' in genre:
        anime.at[i,'Demons']=1
        
    if 'Drama' in genre:
        anime.at[i,'Drama']=1
        
    if 'Ecchi' in genre:
        anime.at[i,'Ecchi']=1
        
    if 'Fantasy' in genre:
        anime.at[i,'Fantasy']=1
        
    if 'Game' in genre:
        anime.at[i,'Game']=1
        
    if 'Harem' in genre:
        anime.at[i,'Harem']=1
        
    if 'Hentai' in genre:
        anime.at[i,'Hentai']=1
        
    if 'Historical' in genre:
        anime.at[i,'Historical']=1
        
    if 'Horror' in genre:
        anime.at[i,'Horror']=1
        
    if 'Josei' in genre:
        anime.at[i,'Josei']=1
        
    if 'Kids' in genre:
        anime.at[i,'Kids']=1
        
    if 'Magic' in genre:
        anime.at[i,'Magic']=1
        
    if 'Martial Arts' in genre:
        anime.at[i,'Martial Arts']=1
        
    if 'Mecha' in genre:
        anime.at[i,'Mecha']=1
        
    if 'Military' in genre:
        anime.at[i,'Military']=1
        
    if 'Music' in genre:
        anime.at[i,'Music']=1
        
    if 'Mystery' in genre:
        anime.at[i,'Mystery']=1
        
    if 'Parody' in genre:
        anime.at[i,'Parody']=1
        
    if 'Police' in genre:
        anime.at[i,'Police']=1
        
    if 'Psychological' in genre:
        anime.at[i,'Psychological']=1
        
    if 'Romance' in genre:
        anime.at[i,'Romance']=1
        
    if 'Samurai' in genre:
        anime.at[i,'Samurai']=1
        
    if 'School' in genre:
        anime.at[i,'School']=1
        
    if 'Sci-Fi' in genre:
        anime.at[i,'Sci-Fi']=1
        
    if 'Seinen' in genre:
        anime.at[i,'Seinen']=1
        
    if 'Shoujo' in genre:
        anime.at[i,'Shoujo']=1
        
    if 'Shoujo Ai' in genre:
        anime.at[i,'Shoujo Ai']=1
        
    if 'Shounen' in genre:
        anime.at[i,'Shounen']=1
        
    if 'Shounen Ai' in genre:
        anime.at[i,'Shounen Ai']=1
        
    if 'Slice of Life' in genre:
        anime.at[i,'Slice of Life']=1
        
    if 'Space' in genre:
        anime.at[i,'Space']=1
        
    if 'Sports' in genre:
        anime.at[i,'Sports']=1
        
    if 'Super Power' in genre:
        anime.at[i,'Super Power']=1
        
    if 'Supernatural' in genre:
        anime.at[i,'Supernatural']=1
        
    if 'Thriller' in genre:
        anime.at[i,'Thriller']=1
        
    if 'Vampire' in genre:
        anime.at[i,'Vampire']=1
        
    if 'Yaoi' in genre:
        anime.at[i,'Yaoi']=1
        
    if 'Yuri' in genre:
        anime.at[i,'Yuri']=1

anime = anime.drop('genre',axis=1)

#Forming clusters using K Means Clustering (unsupervised machine learning)
data = anime[['episodes','score', 'Action', 'Adventure', 'Cars',
       'Comedy', 'Dementia', 'Demons', 'Drama', 'Ecchi', 'Fantasy', 'Game',
       'Harem', 'Hentai', 'Historical', 'Horror', 'Josei', 'Kids', 'Magic',
       'Martial Arts', 'Mecha', 'Military', 'Music', 'Mystery', 'Parody',
       'Police', 'Psychological', 'Romance', 'Samurai', 'School', 'Sci-Fi',
       'Seinen', 'Shoujo', 'Shoujo Ai', 'Shounen', 'Shounen Ai',
       'Slice of Life', 'Space', 'Sports', 'Super Power', 'Supernatural',
       'Thriller', 'Vampire', 'Yaoi', 'Yuri']]

data = data.drop('episodes',axis=1)
data.dropna(subset=['score'],inplace=True)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=500)
kmeans.fit(data)

cluster = pd.DataFrame(kmeans.labels_,columns=['Cluster'])

#Formatting data
anime = anime.dropna(subset=['score'])
anime = anime.reset_index()
del anime['index']

df = pd.concat([anime,cluster],axis=1)

#execution
anime_name = input('Please enter the name of an anime you like and we will recommend more of the same type!: ')
chosen_cluster = int(df[df['title']==anime_name]['Cluster'])

print(df[df['Cluster']==chosen_cluster]['title'])

input('Press ENTER to exit')


