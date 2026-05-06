import torch
import numpy as np
import pickle
import pandas as pd

from src.models.transformer import TransformerModel

SEQ_LEN = 10

class InferenceEngine:

    def __init__(self):

        self.movies_df = pd.read_csv(
            "data/movies_processed.csv"
        )

        with open(
            "saved_models/movie_encoder.pkl",
            "rb"
        ) as f:

            self.encoder = pickle.load(f)

        self.num_items = len(
            self.encoder.classes_
        )

        self.model = TransformerModel(
            num_items=self.num_items
        )

        self.model.load_state_dict(
            torch.load(
                "saved_models/transformer.pth",
                map_location="cpu"
            )
        )

        self.model.eval()

    def recommend(self, watched_titles, top_k=10):

        watched_movies = self.movies_df[
            self.movies_df["title"].isin(
                watched_titles
            )
        ]

        if len(watched_movies) == 0:
            return []

        watched_ids = watched_movies[
            "movieId_enc"
        ].tolist()

        # =============================================
        # PAD SEQUENCE
        # =============================================

        if len(watched_ids) < SEQ_LEN:

            pad_len = SEQ_LEN - len(watched_ids)

            watched_ids = (
                [0] * pad_len
                + watched_ids
            )

        else:
            watched_ids = watched_ids[-SEQ_LEN:]

        seq = torch.LongTensor(
            [watched_ids]
        )

        with torch.no_grad():

            logits, attention = self.model(seq)

            probs = torch.softmax(
                logits,
                dim=-1
            )

        probs = probs.numpy()[0]

        top_indices = np.argsort(probs)[::-1]

        recommendations = []

        for idx in top_indices:

            movie_row = self.movies_df[
                self.movies_df["movieId_enc"] == idx
            ]

            if len(movie_row) == 0:
                continue

            title = movie_row.iloc[0]["title"]

            if title not in watched_titles:

                recommendations.append({
                    "title": title,
                    "score": float(probs[idx])
                })

            if len(recommendations) >= top_k:
                break

        return recommendations, attention