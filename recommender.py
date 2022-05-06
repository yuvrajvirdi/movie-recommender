from re import X
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval

# making dataframes
credits_df = pd.read_csv("tmdb_5000_credits.csv")
movies_df = pd.read_csv("tmdb_5000_movies.csv")

# merging dataframes
credits_df.columns = ["id", "tittle", "cast", "crew"]
movies_df = movies_df.merge(credits_df, on="id")

# filtering by demographic
C = movies_df["vote_average"].mean()
m = movies_df["vote_count"].quantile(0.9)

curr_movies_df = movies_df.copy().loc[movies_df["vote_count"] >= m]

# weighing ratings
def weighted_ratings(x, C=C, m=m):
    v = x["vote_count"]
    R = x["vote_average"]
    return (v/(v+m)*R)+(m/(v+m)*C)

curr_movies_df["score"] = curr_movies_df.apply(weighted_ratings, axis=1)
curr_movies_df = curr_movies_df.sort_values("score", ascending=False)

# filtering by content
tfidf = TfidfVectorizer(stop_words="english")
movies_df["overview"] = movies_df["overview"].fillna("") # fill N/A

tfidf_matrix = tfidf.fit_transform(movies_df["overview"])

# similarity computation
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies_df.index, index=movies_df["title"]).drop_duplicates()

# function to get movies
def get_recommendations(title, cosine_sim=cosine_sim):
    i = indices[title]
    sim_scores = list(enumerate(cosine_sim[i]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11] # top 10
    movies_indices = [idx[0] for idx in sim_scores]
    movies = movies_df["title"].iloc[movies_indices]
    return movies.array

# metadata based search
features = ["cast", "crew", "keywords", "genres"]
for feature in features:
    movies_df[feature] = movies_df[feature].apply(literal_eval)

def get_director(x):
    for i in x:
        if i["job"] == "Director":
            return i["name"]
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]
        if len(names)>3:
            names = names[:3]
        return names
    return []

movies_df["director"] = movies_df["crew"].apply(get_director)

features = ["cast", "keywords", "genres"]
for feature in features:
    movies_df[feature] = movies_df[feature].apply(get_list)

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ""

features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    movies_df[feature] = movies_df[feature].apply(clean_data)

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

movies_df["soup"] = movies_df.apply(create_soup, axis=1)

count_vectorizer = CountVectorizer(stop_words="english")
count_matrix = count_vectorizer.fit_transform(movies_df["soup"])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

movies_df = movies_df.reset_index()
indices = pd.Series(movies_df.index, index=movies_df['title'])
