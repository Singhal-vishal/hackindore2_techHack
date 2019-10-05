from flask import Flask, request
import json
from flask_cors import CORS, cross_origin
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from fuzzywuzzy import fuzz
from sklearn.neighbors import NearestNeighbors
import pickle
from File1 import movie_user_mat_sparse,movie_to_idx, df_movies, df_ratings

app = Flask(__name__)
cors = CORS(app)

model_knn = pickle.load(open('model.sav', 'rb'))
def fuzzy_matching(mapper, fav_movie, verbose=True):
    match_tuple = []
    # get match


    for title, idx in mapper.items():
        ratio = fuzz.ratio(title.lower(), fav_movie.lower())
        if ratio >= 60:
            match_tuple.append((title, idx, ratio))
    # sort
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        print('Oops! No match is found')
        return None
    if verbose:
        print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][1]


@app.route("/recommend")
@cross_origin()
def make_recommendation():
    fav_movie = request.args.get('moviename')  
    # fav_movie = 'Iron Man'  
    # fit
    model_knn.fit(movie_user_mat_sparse)

    # get input movie index
    print('You have input movie:', fav_movie)
    idx = fuzzy_matching(movie_to_idx, fav_movie, verbose=True)
    if idx is None:
        return 'Oops! No match is found'
    # inference
    print('Recommendation system start to make inference')
    print('......\n')
    distances, indices = model_knn.kneighbors(movie_user_mat_sparse[idx], n_neighbors=10+1)
    # get list of raw idx of recommendations
    raw_recommends = \
        sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    # get reverse mapper
    reverse_mapper = {v: k for k, v in movie_to_idx.items()}
    # print recommendations
    # print('Recommendations for {}:'.format(fav_movie))
    list22 = []
    for i, (idx, dist) in enumerate(raw_recommends):

        list22.append('{0}: {1}'.format(i+1, reverse_mapper[idx]))
    print(list22)
    return json.dumps(list22)

# model_knn, data, mapper, fav_movie, n_recommendations
# make_recommendation(
#     model_knn=model_knn,
#     data=movie_user_mat_sparse,
#     fav_movie=my_favorite,
#     mapper=movie_to_idx,
#     n_recommendations=10)

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0')