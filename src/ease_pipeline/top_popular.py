import pickle

import pandas as pd


class TopPopular:
    def __init__(self):
        self.recommendations = None
        self.trained = False

    def fit(self, df: pd.DataFrame, item_col: str = "item_id") -> None:
        self.recommendations = df[item_col].value_counts().index.to_numpy()
        self.trained = True

    def predict(self, top_k: int = 20) -> list:
        if not self.trained:
            raise RuntimeError("You must fit the model before making predictions.")

        top_k_recs = self.recommendations[:top_k].tolist()
        return top_k_recs

    def save(self, filepath: str) -> None:
        """Сохранить модель в файл"""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "TopPopular":
        """Загрузить модель из файла"""
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model
