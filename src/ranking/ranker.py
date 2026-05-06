import lightgbm as lgb
import numpy as np
import pickle

class Ranker:
    def __init__(self):
        self.model = lgb.LGBMRanker(objective="lambdarank")

    def train(self, X, y, group):
        self.model.fit(X, y, group=group)

        with open("saved_models/ranker.pkl", "wb") as f:
            pickle.dump(self.model, f)

    def load(self):
        with open("saved_models/ranker.pkl", "rb") as f:
            self.model = pickle.load(f)

    def predict(self, X):
        return self.model.predict(X)