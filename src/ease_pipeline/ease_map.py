"""
EASE Model Wrapper with Name Mapping
Provides recommendation interface that works with ingredient names
"""

import pickle
from pathlib import Path
from typing import List, Union

import numpy as np


class EASERecommenderWithNames:
    """
    EASE Recommender that can work with both item IDs and ingredient names.

    Usage:
        # Initialize
        recommender = EASERecommenderWithNames(
            model_path='../recsys_tests/ease_model.pkl',
            names_path='../recsys_tests/items_dict.json'
        )

        # Get recommendations by item IDs
        recs = recommender.recommend([0, 1, 2], top_k=10)

        # Get recommendations by ingredient names
        recs = recommender.recommend(['Молоко', 'Яйцо куриное', 'Масло сливочное'], top_k=10)

        # Get recommendations with names
        recs = recommender.recommend_with_names(['Молоко', 'Чеснок'], top_k=10)
    """

    def __init__(self, model_path: str, names_path: str):
        """
        Initialize the recommender with model and name mappings.

        Args:
            model_path: Path to the pickled EASE model
            names_path: Path to the JSON file with id->name mappings
        """
        self.model_path = Path(model_path)
        # self.names_path = Path(names_path)

        # Load model
        print(f"Loading EASE model from {self.model_path}...")
        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model_weights = model_data["model_weights"]
        self.item2id = model_data["item2id"]
        self.id2item = model_data["id2item"]
        self.n_items = model_data["n_items"]
        self.n_users = model_data.get("n_users", 0)

        print(f"   Model loaded: {self.n_items} items, {self.n_users} users")
        print(f"   Weights shape: {self.model_weights.shape}")

        # Load name mappings
        # print(f"Loading item names from {self.names_path}...")
        # ic(self.names_path)
        # with open(self.names_path, "r", encoding="utf-8") as f:
        #     self.id2name = json.load(f)

        # Convert string keys to int
        # self.id2name = {int(k): v for k, v in self.id2name.items()}

        # Create reverse mapping: name -> id
        # ic(self.id2name)
        # self.name2id = {v: k for k, v in self.id2name.items()}
        # ic(self.name2id)
        # print(f"   Loaded {len(self.id2name)} item names")

    # def _convert_to_ids(self, items: List[Union[int, str]]) -> List[int]:
    #     """
    #     Convert a list of items (IDs or names) to item IDs.

    #     Args:
    #         items: List of item IDs (int) or ingredient names (str)

    #     Returns:
    #         List of item IDs
    #     """
    #     item_ids = []
    #     for item in items:
    #         if isinstance(item, str):
    #             # Convert name to ID
    #             if item in self.name2id:
    #                 item_ids.append(self.name2id[item])
    #             else:
    #                 print(f"Warning: Item '{item}' not found in mappings")
    #         elif isinstance(item, (int, np.integer)):
    #             # Already an ID
    #             if item in self.id2item or item in self.id2name:
    #                 item_ids.append(int(item))
    #             else:
    #                 print(f"Warning: Item ID {item} not found in model")
    #         else:
    #             print(f"Warning: Unsupported item type: {type(item)}")

    #     return item_ids

    def recommend(
        self,
        user_items: List[Union[int, str]],
        top_k: int = 20,
        exclude_seen: bool = True,
    ) -> List[int]:
        """
        Generate top-K item recommendations based on user's interaction history.

        Args:
            user_items: List of items (IDs or names) the user has interacted with
            top_k: Number of recommendations to return
            exclude_seen: Whether to exclude items the user has already seen

        Returns:
            List of recommended item IDs
        """
        # Convert to IDs
        # item_ids = self._convert_to_ids(user_items)
        # ic(item_ids)
        if not user_items:
            print(
                "Warning: No valid items provided, returning empty recommendations, NEED TO APPLY TOPPOP"
            )
            return []

        # Create user vector
        user_vector = np.zeros(self.n_items, dtype=np.float32)
        user_vector[user_items] = 1.0

        # Compute scores
        scores = user_vector @ self.model_weights

        # Exclude already seen items
        if exclude_seen:
            scores[user_items] = -np.inf

        # Get top-K items
        top_indices = np.argsort(-scores)[:top_k]

        return [int(idx) for idx in top_indices]

    # NOTE: Unused
    # def recommend_with_names(
    #     self,
    #     user_items: List[Union[int, str]],
    #     top_k: int = 20,
    #     exclude_seen: bool = True,
    # ) -> List[tuple]:
    #     """
    #     Generate top-K recommendations with both IDs and names.

    #     Args:
    #         user_items: List of items (IDs or names) the user has interacted with
    #         top_k: Number of recommendations to return
    #         exclude_seen: Whether to exclude items the user has already seen

    #     Returns:
    #         List of tuples (item_id, item_name, score)
    #     """
    #     # Convert to IDs
    #     item_ids = self._convert_to_ids(user_items)

    #     if not user_items:
    #         print("Warning: No valid items provided, returning empty recommendations, NEED TO APPLY TOPPOP")
    #         return []

    #     # Create user vector
    #     user_vector = np.zeros(self.n_items, dtype=np.float32)
    #     user_vector[user_items] = 1.0

    #     # Compute scores
    #     scores = user_vector @ self.model_weights

    #     # Exclude already seen items
    #     if exclude_seen:
    #         scores[user_items] = -np.inf

    #     # Get top-K items
    #     top_indices = np.argsort(-scores)[:top_k]

    #     # Build result with names and scores
    #     results = []
    #     for idx in top_indices:
    #         item_id = int(idx)
    #         item_name = self.id2name.get(item_id, f"Unknown_{item_id}")
    #         score = float(scores[idx])
    #         results.append((item_id, item_name, score))

    #     return results

    # def get_item_name(self, item_id: int) -> str:
    #     """Get the name of an item by its ID."""
    #     return self.id2name.get(item_id, f"Unknown_{item_id}")

    # def get_item_id(self, item_name: str) -> int:
    #     """Get the ID of an item by its name."""
    #     if item_name in self.name2id:
    #         return self.name2id[item_name]
    #     raise ValueError(f"Item name '{item_name}' not found")

    def batch_recommend(
        self,
        user_items_list: List[List[Union[int, str]]],
        top_k: int = 20,
        exclude_seen: bool = True,
    ) -> List[List[int]]:
        """
        Generate recommendations for multiple users at once.

        Args:
            user_items_list: List of interaction lists for each user
            top_k: Number of recommendations per user
            exclude_seen: Whether to exclude already seen items

        Returns:
            List of recommendation lists (one per user)
        """
        return [self.recommend(items, top_k, exclude_seen) for items in user_items_list]

    # def batch_recommend_with_names(
    #     self,
    #     user_items_list: List[List[Union[int, str]]],
    #     top_k: int = 20,
    #     exclude_seen: bool = True,
    # ) -> List[List[tuple]]:
    #     """
    #     Generate recommendations with names for multiple users at once.

    #     Args:
    #         user_items_list: List of interaction lists for each user
    #         top_k: Number of recommendations per user
    #         exclude_seen: Whether to exclude already seen items

    #     Returns:
    #         List of recommendation lists with names (one per user)
    #     """
    #     return [
    #         self.recommend_with_names(items, top_k, exclude_seen)
    #         for items in user_items_list
    #     ]
