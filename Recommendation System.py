#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 23:36:32 2019
Title:Collaborative Filtering using movie dataset
"""

import pandas as pd 
import numpy as np


ratings_data = pd.read_csv("ratings.csv")
#print(ratings_data.head())

#function to return ratings data to 3-tuple
def buildMatrix(df):
    ###
    ### YOUR CODE HERE
    df = df.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
    ratings = df.as_matrix()
    user_id_index = dict(zip(list(df.index), list(range(len(df.index))))) 
    #print(user_id_index)
    index_movie_id = dict(zip(list(range(len(df.columns.values))), list(df.columns.values)))
    return ratings, user_id_index, index_movie_id


#function to compute cosine similarity
def cosine(vA,vB):
    ###
    ### YOUR CODE HERE
    dot_product = np.dot(vA, vB)
    norm_a = np.linalg.norm(vA)
    norm_b = np.linalg.norm(vB)
    return dot_product/(norm_a*norm_b)


#function to choose top 10 similar users and return userId
def select_users(ratings, user_id_index, current_user_id):
    similarity = []
    current_user_profile = ratings[users_id_index[current_user_id]]
    
    for u_id in user_id_index.keys():
        if current_user_id != u_id:
            u_id_profile = ratings[user_id_index[u_id]]
            similarity.append((u_id, cosine(u_id_profile,
                                            current_user_profile)))
    
    similarity.sort(key = lambda x:x[1], reverse=True)
    return [x[0] for x in similarity[0:10]]


#function to recommend based on preference of users through user method
def recommend_user_user(ratings, similar_user_ids, user_id_index, index_movie_id):
    recommendations = set()
    for similar_user_id in similar_user_ids:
        similar_user_profile = ratings[user_id_index[similar_user_id]]
        # print(similar_user_profile)
        for indx,movie_rating in enumerate(similar_user_profile):
            if movie_rating!=0.0:
                recommendations.add(index_movie_id[indx])
                break 
    return recommendations 


