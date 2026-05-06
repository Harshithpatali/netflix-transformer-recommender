import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ContentModel:
    def __init__(self, data_path="data"):
        self.movies = pd.read_csv(f"{data_path}/movies_processed.csv")

        genres = self.movies["genres"].str.get_dummies(sep="|")
        self.genre_matrix = genres.values

        self.similarity = cosine_similarity(self.genre_matrix)

    def get_similarity(self, movie_idx):
        return self.similarity[movie_idx]