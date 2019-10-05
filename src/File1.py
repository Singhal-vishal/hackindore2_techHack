import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from fuzzywuzzy import fuzz
from sklearn.neighbors import NearestNeighbors
import pickle

movies_filename = 'movies.csv'
ratings_filename ='ratings.csv'

df_movies = pd.read_csv(
	movies_filename,
	usecols=['movieId', 'title'],
	dtype={'movieId':'int32','title':'str'})

# print("DF Movies")
# print(df_movies.head())

df_ratings = pd.read_csv(
	ratings_filename,
	usecols = ['userId', 'movieId', 'rating'],
	dtype = {'userId':'int32', 'movieId':'int32', 'rating':'float32'})

# print("DF Rating")
# print(df_ratings.head())

num_users = len(df_ratings.userId.unique())
num_items = len(df_ratings.movieId.unique())

#get count of particular rating
df_ratings_cnt = pd.DataFrame(df_ratings.groupby('rating').size(), columns=['count'])
# print(df_ratings_cnt)

#there are a lot more counts in rating of zero
total_cnt = num_items*num_users
rating_zero_cnt = total_cnt - df_ratings.shape[0]

#append counts of zero rating to df_ratings_cnt
df_ratings_cnt = df_ratings_cnt.append(pd.DataFrame({'count':rating_zero_cnt}, index = [0.0]),
	verify_integrity=True).sort_index()
# print(df_ratings_cnt)


#taking log count of rating
df_ratings_cnt['log_count'] = np.log(df_ratings_cnt['count'])
#print(df_ratings_cnt)
ax = df_ratings_cnt[['count']].reset_index().rename(columns={'index':'rating score'}).plot(x='rating score',y='count',kind='bar',figsize=(12,8),title='count for each rating',logy=True,fontsize=12)
ax.set_xlabel("movie rating score")
ax.set_ylabel("number of rating")
# plt.show()
df_movies_cnt = pd.DataFrame(df_ratings.groupby('movieId').size(), columns=['count'])
# filter data
popularity_thres = 10000
popular_movies = list(set(df_movies_cnt.query('count >= @popularity_thres').index))
df_ratings_drop_movies = df_ratings[df_ratings.movieId.isin(popular_movies)]
#print('shape of original ratings data: ', df_ratings.shape)
#print('shape of ratings data after dropping unpopular movies: ', df_ratings_drop_movies.shape)
#print(df_movies_cnt)

# get number of ratings given by every user
df_users_cnt = pd.DataFrame(df_ratings_drop_movies.groupby('userId').size(), columns=['count'])

ratings_thres = 100
active_users = list(set(df_users_cnt.query('count >= @ratings_thres').index))
df_ratings_drop_users = df_ratings_drop_movies[df_ratings_drop_movies.userId.isin(active_users)]
#print('shape of original ratings data: ', df_ratings.shape)
#print('shape of ratings data after dropping both unpopular movies and inactive users: ', df_ratings_drop_users.shape)

#pivote and create movie-user matrix
movie_user_mat = df_ratings_drop_users.pivot(index = 'movieId',
					columns='userId', values = 'rating').fillna(0)

# print(movie_user_mat)

#create mapper from movie title to index
movie_to_idx = {
	movie : i for i, movie in
	enumerate(list(df_movies.set_index('movieId')
		.loc[movie_user_mat.index].title))
	}

# print(movie_to_idx)

#transform matrix to scipy sparse matrix
movie_user_mat_sparse = csr_matrix(movie_user_mat.values)

# %env JOBLIB_TEMP_FOLDER=/tmp
# define model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
# fit
model_knn.fit(movie_user_mat_sparse)

filename = 'model.sav'
pickle.dump(model_knn,open(filename,'wb'))

#print(model_knn)
