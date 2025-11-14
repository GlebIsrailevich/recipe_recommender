import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from scipy import sparse as sps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EASERecommender:
    """
    EASE (Embarrassingly Shallow Autoencoders) Recommender System

    A simple yet effective collaborative filtering model that can be used
    to recommend items based on user interaction history.

    Attributes:
        model_weights (np.ndarray): Trained EASE model weights
        item2id (Dict): Mapping from item identifiers to internal IDs
        id2item (Dict): Mapping from internal IDs to item identifiers
        user2id (Dict): Mapping from user identifiers to internal IDs
        id2user (Dict): Mapping from internal IDs to user identifiers
        n_items (int): Total number of items in the catalog
        n_users (int): Total number of users
    """

    def __init__(self):
        """Initialize the EASE recommender."""
        self.model_weights: Optional[np.ndarray] = None
        self.item2id: Dict = {}
        self.id2item: Dict = {}
        self.user2id: Dict = {}
        self.id2user: Dict = {}
        self.n_items: int = 0
        self.n_users: int = 0
        self._is_trained: bool = False

    def fit(
        self,
        user_ids: Union[List, np.ndarray],
        item_ids: Union[List, np.ndarray],
        reg_weight: float = 100.0,
    ) -> "EASERecommender":
        """
        Train the EASE model on user-item interactions.

        Args:
            user_ids: Array of user identifiers
            item_ids: Array of item identifiers (same length as user_ids)
            reg_weight: Regularization weight for the model (default: 100.0)

        Returns:
            self: The trained recommender instance

        Raises:
            ValueError: If user_ids and item_ids have different lengths
        """
        if len(user_ids) != len(item_ids):
            raise ValueError("user_ids and item_ids must have the same length")

        logger.info("Building mappings...")
        # Create mappings
        unique_users = np.unique(user_ids)
        unique_items = np.unique(item_ids)

        self.user2id = {user: idx for idx, user in enumerate(unique_users)}
        self.id2user = {idx: user for user, idx in self.user2id.items()}
        self.item2id = {item: idx for idx, item in enumerate(unique_items)}
        self.id2item = {idx: item for item, idx in self.item2id.items()}

        self.n_users = len(self.user2id)
        self.n_items = len(self.item2id)

        logger.info(
            f"Users: {self.n_users}, Items: {self.n_items}, Interactions: {len(user_ids)}"
        )

        # Encode IDs
        encoded_users = np.array([self.user2id[u] for u in user_ids])
        encoded_items = np.array([self.item2id[i] for i in item_ids])

        # Create interaction matrix
        logger.info("Creating interaction matrix...")
        interaction_matrix = sps.coo_matrix(
            (np.ones(len(encoded_users)), (encoded_users, encoded_items)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32,
        ).tocsr()

        # Train EASE model
        logger.info("Training EASE model...")
        self.model_weights = self._fit_ease(interaction_matrix, reg_weight)
        self._is_trained = True

        logger.info("Training complete!")
        return self

    def _fit_ease(self, X: sps.csr_matrix, reg_weight: float = 100.0) -> np.ndarray:
        """
        Internal method to fit the EASE model.

        Args:
            X: User-item interaction matrix (users x items)
            reg_weight: Regularization weight

        Returns:
            Trained model weight matrix
        """
        # Gram matrix (item-item similarity)
        G = X.T @ X

        # Add regularization to diagonal
        G += reg_weight * sps.identity(G.shape[0])

        # Convert to dense (inverse will be dense anyway)
        G = G.todense()

        # Compute inverse
        logger.info("Computing matrix inverse...")
        P = np.linalg.inv(G)

        # Compute final weights
        B = P / (-np.diag(P))

        # Zero out diagonal (no self-recommendations)
        np.fill_diagonal(B, 0.0)

        return np.asarray(B)

    def recommend(
        self,
        #   user_id: Union[str, int],
        user_interactions: List[Union[str, int]],
        top_k: int = 20,
        exclude_seen: bool = True,
    ):
        """
        Generate top-K recommendations for a user based on their interaction history.

        Args:
            user_id: Identifier of the user (can be unused, kept for API consistency)
            user_interactions: List of items the user has interacted with
            top_k: Number of recommendations to return (default: 20)
            exclude_seen: Whether to exclude already seen items (default: True)

        Returns:
            List of recommended item identifiers

        Raises:
            RuntimeError: If model hasn't been trained yet
            ValueError: If top_k is less than 1
        """
        if not self._is_trained:
            raise RuntimeError(
                "Model must be trained before making recommendations. Call fit() first."
            )

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        # Encode user interactions
        encoded_interactions = []
        for item in user_interactions:
            if item in self.item2id:
                encoded_interactions.append(self.item2id[item])

        if not encoded_interactions:
            logger.warning(
                "No valid items found in user interactions. Returning popular items."
            )
            # Return top items by row sum (popularity)
            item_scores = self.model_weights.sum(axis=0)
            top_indices = np.argsort(-item_scores)[:top_k]
            return [int(self.id2item[idx]) for idx in top_indices]

        # Create user vector
        user_vector = np.zeros(self.n_items, dtype=np.float32)
        user_vector[encoded_interactions] = 1.0

        # Compute scores
        scores = user_vector @ self.model_weights

        # Exclude already seen items
        if exclude_seen:
            scores[encoded_interactions] = -np.inf

        # Get top-K items
        top_indices = np.argsort(-scores)[:top_k]

        # Decode and return
        recommendations = [int(self.id2item[idx]) for idx in top_indices]

        return recommendations

    def recommend_batch(
        self,
        user_interactions_list: List[List[Union[str, int]]],
        top_k: int = 20,
        exclude_seen: bool = True,
    ) -> List[List[Union[str, int]]]:
        """
        Generate recommendations for multiple users at once.

        Args:
            user_interactions_list: List of interaction lists for each user
            top_k: Number of recommendations per user
            exclude_seen: Whether to exclude already seen items

        Returns:
            List of recommendation lists (one per user)
        """
        return [
            self.recommend(None, interactions, top_k, exclude_seen)
            for interactions in user_interactions_list
        ]

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the trained model to disk.

        Args:
            path: File path to save the model
        """
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model_weights": self.model_weights,
            "item2id": self.item2id,
            "id2item": self.id2item,
            "user2id": self.user2id,
            "id2user": self.id2user,
            "n_items": self.n_items,
            "n_users": self.n_users,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EASERecommender":
        """
        Load a trained model from disk.

        Args:
            path: File path to load the model from

        Returns:
            Loaded EASERecommender instance
        """
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        recommender = cls()
        recommender.model_weights = model_data["model_weights"]
        recommender.item2id = model_data["item2id"]
        recommender.id2item = model_data["id2item"]
        recommender.user2id = model_data["user2id"]
        recommender.id2user = model_data["id2user"]
        recommender.n_items = model_data["n_items"]
        recommender.n_users = model_data["n_users"]
        recommender._is_trained = True

        logger.info(f"Model loaded from {path}")
        return recommender

    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return self._is_trained

    def get_similar_items(
        self, item_id: Union[str, int], top_k: int = 10
    ) -> List[tuple]:
        """
        Find items similar to a given item.

        Args:
            item_id: Item identifier
            top_k: Number of similar items to return

        Returns:
            List of (item_id, similarity_score) tuples
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained first")

        if item_id not in self.item2id:
            raise ValueError(f"Item {item_id} not found in training data")

        item_idx = self.item2id[item_id]
        similarities = self.model_weights[:, item_idx]

        # Get top-K (excluding the item itself)
        top_indices = np.argsort(-similarities)[: top_k + 1]
        top_indices = top_indices[top_indices != item_idx][:top_k]

        similar_items = [
            (int(self.id2item[idx]), float(similarities[idx])) for idx in top_indices
        ]

        return similar_items
