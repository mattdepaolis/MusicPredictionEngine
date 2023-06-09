#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 08:34:57 2023

References: 
https://making.lyst.com/lightfm/docs/home.html
https://msha096.github.io/blog/lightfm/
https://github.com/V-Sher/LightFm_HybridRecommenderSystem/blob/master/LightFM%20Worked%20Example.ipynb

@author: Matthias & Tejesh
"""
from lightfm import LightFM, cross_validation
from lightfm.evaluation import auc_score, precision_at_k
from lightfm.data import Dataset
import pandas as pd
import numpy as np

# Preprocessing dataset
# Load the dataset
data = pd.read_csv('train.csv')
# clean data
df_train = data.dropna(axis='rows')
print("NA values: ", df_train.isna().sum())
data = data.dropna(axis='rows')
# change values 0 to -1
data['is_listened'] = data['is_listened'].replace(to_replace=0, value=-1)


# Create a new Dataset object
dataset = Dataset()

# Map the user and item IDs to unique integer values
dataset.fit(data['user_id'], data['media_id'])

num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items {}.'.format(num_users, num_items))

df_main = data[['user_id', 'media_id', 'is_listened']]
user_df= data[['user_id', 'user_age', 'user_gender']]
item_df = data[['genre_id', 'ts_listen', 'media_id', 'album_id', 'context_type', 'media_duration', 'artist_id']]


# Create user features
user_features = []
user_columns = ['user_gender'] * len(user_df.user_gender.unique()) + ['user_age'] * len(user_df.user_age.unique())
unique_features_user = list(user_df.user_gender.unique()) + list(user_df.user_age.unique())

for x,y in zip(user_columns, unique_features_user):
    res = str(x)+ ":" +str(y)
    user_features.append(res)
    #print(res)
    
    
# Create item features
item_features = []
item_columns = ['genre_id'] * len(item_df.genre_id.unique()) + ['ts_listen'] * len(item_df.ts_listen.unique()) + \
    ['album_id'] * len(item_df.album_id.unique()) + ['context_type'] * len(item_df.context_type.unique()) + \
        ['media_duration'] * len(item_df.media_duration.unique()) + ['artist_id'] * len(item_df.artist_id.unique())
unique_features_item = list(item_df.genre_id.unique()) + list(item_df.ts_listen.unique()) + \
    list(item_df.album_id.unique()) + list(item_df.context_type.unique()) + \
        list(item_df.media_duration.unique()) + list(item_df.artist_id.unique()) 
        
for x,y in zip(item_columns, unique_features_item):
    res = str(x)+ ":" +str(y)
    item_features.append(res)
    #print(res)

# Fitting the dataset
dataset.fit(users=df_main['user_id'].unique(), 
                    items=df_main['media_id'].unique(),
                    user_features=(user_features),
                    item_features=(item_features))

# plugging in the interactions and their weights
(interactions, weights) = dataset.build_interactions([(x[0], x[1], x[2]) for x in df_main.values])
print(repr(interactions))

print(interactions.todense())
print(weights.todense())


# Building user features
def feature_colon_value_user(my_list):
    result = []
    ll = ['user_gender:', 'user_age:']
    aa = my_list
    for x,y in zip(ll,aa):
        res = str(x) +""+ str(y)
        result.append(res)
    return result

ad_subset = user_df[['user_gender','user_age']] 
ad_list = [list(x) for x in ad_subset.values]
feature_list_user = []
for user in ad_list:
    feature_list_user.append(feature_colon_value_user(user))
    #print(feature_colon_value_user(user))
#print(f'Final output: {feature_list}') 
user_tuple = list(zip(df_main.user_id, feature_list_user))
user_features = dataset.build_user_features(user_tuple, normalize= True)

# Building item features
def feature_colon_value_item(my_list):
    result = []
    ll = ['genre_id:', 'ts_listen:','album_id:', 'context_type:', 'media_duration:', 'artist_id:']
    aa = my_list
    for x,y in zip(ll,aa):
        res = str(x) +""+ str(y)
        result.append(res)
    return result

ad_subset = item_df[['genre_id', 'ts_listen','album_id', 'context_type', 'media_duration', 'artist_id']]
ad_list = [list(x) for x in ad_subset.values]
feature_list_item = []
for item in ad_list:
    feature_list_item.append(feature_colon_value_item(item))
    #print(feature_colon_value_item(item))
#print(f'Final output: {feature_list}') 
item_tuple = list(zip(df_main.media_id, feature_list_item))
item_features = dataset.build_item_features(item_tuple, normalize= True)

# Split data into training test set
train, test = cross_validation.random_train_test_split(interactions, test_percentage=0.2, random_state=np.random.RandomState(42))
train_weights, test_weights = cross_validation.random_train_test_split(interactions, test_percentage=0.2, random_state=np.random.RandomState(42))


# Training the model
# pure CF
model_cf = LightFM(loss = 'warp',
                   no_components = 160,
                   item_alpha = 1e-7,
                   learning_rate = 0.02,
                   max_sampled = 50)

model_cf.fit(train, 
             sample_weight=train_weights, 
             epochs = 50, 
             num_threads = 4)

# hybrid model
model_hybrid = LightFM(loss = 'warp',
                no_components = 160,
                item_alpha = 1e-7,
                learning_rate = 0.02,
                max_sampled = 50)

model_hybrid.fit(train, 
                 sample_weight=train_weights, 
                 user_features=user_features,
                 item_features = item_features, 
                 epochs = 50, 
                 num_threads = 4)


# Evaluation
df_result = pd.DataFrame(columns = ['Method', 'Evaluation Metric', 'Train', 'Test'])

# pure CF model
auc_train = auc_score(model_cf, train).mean()
auc_test = auc_score(model_cf, test).mean()
print(auc_train, auc_test)

precision_train = precision_at_k(model_cf, train, k = 10).mean()
precision_test = precision_at_k(model_cf, test, k = 10).mean()
precision_train, precision_test 

df_result = df_result.append(pd.DataFrame([['Pure CF', 'AUC', auc_train, auc_test],
                                           ['Pure CF', 'Precision@10', precision_train, precision_test]],
                                          columns = df_result.columns))

# hybrid model
auc_train = auc_score(model_hybrid, train, item_features = item_features, user_features=user_features).mean()
auc_test = auc_score(model_hybrid, test, item_features = item_features, user_features=user_features).mean()
auc_train, auc_test

precision_train = precision_at_k(model_cf, train, k = 10).mean()
precision_test = precision_at_k(model_cf, test, k = 10).mean()
print(precision_train, precision_test )

df_result = df_result.append(pd.DataFrame([['Hybrid model', 'AUC', auc_train, auc_test],
                                           ['Hybrid model', 'Precision@10', precision_train, precision_test]],
                                          columns = df_result.columns))

df_result.sort_values(by = ['Evaluation Metric'])
