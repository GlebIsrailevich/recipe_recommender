"""
Main EASE Recommender Interface
Simple API for getting ingredient recommendations from user activity
"""

import pickle
from typing import List, Tuple

from config import EASEMODEL, MAPPING, TOPPOPULAR
from ease_map import EASERecommenderWithNames
from input_map import IngredientMapper
from top_popular import TopPopular  # noqa: F401 - needed for pickle


class EASERecommendationSystem:
    """
    Main interface for EASE recommendations.
    Takes ingredient names as input and returns ingredient names as output.

    Usage:
        # Initialize the system
        system = EASERecommendationSystem()

        # Get recommendations from user activity (ingredient names)
        user_activity = ['Молоко', 'Яйцо куриное', 'Масло сливочное']
        recommendations = system.get_recommendations(user_activity, top_k=10)

        print(f"Based on: {user_activity}")
        print(f"We recommend: {recommendations}")
    """

    def __init__(
        self,
        model_path: str = EASEMODEL,
        names_path: str = MAPPING,
        toppop_path: str = TOPPOPULAR,
    ):
        """
        Initialize the recommendation system.

        Args:
            model_path: Path to the EASE model pickle file
            names_path: Path to the ingredient names JSON file
            toppop_path: Path to the top popular model pickle file
        """
        print("Initializing EASE Recommendation System...")
        print("-" * 70)

        # Load ingredient mapper
        self.mapper = IngredientMapper(names_path)

        # Load EASE recommender
        self.recommender = EASERecommenderWithNames(
            model_path=model_path, names_path=names_path
        )

        # Load top popular model as fallback
        print(f"Loading Top Popular fallback from {toppop_path}...")
        with open(toppop_path, "rb") as f:
            self.top_popular = pickle.load(f)
        print(f"   Top Popular loaded: {len(self.top_popular.recommendations)} items")

        print("-" * 70)
        print("System ready!\n")

    def get_recommendations(
        self, user_activity: List[str], top_k: int = 10, exclude_seen: bool = True
    ) -> List[str]:
        """
        Get ingredient recommendations based on user activity.
        Uses top popular items as fallback if EASE returns insufficient recommendations.

        Args:
            user_activity: List of ingredient names the user has
            top_k: Number of recommendations to return
            exclude_seen: Whether to exclude ingredients user already has

        Returns:
            List of recommended ingredient names (exactly top_k items)
        """
        # Convert names to IDs
        item_ids = self.mapper.names_to_ids(
            user_activity, skip_unknown=True, warn_unknown=False
        )

        # If no valid items, use top popular
        if not item_ids:
            return self._get_top_popular_names(top_k, exclude_ids=[])

        # Get recommendations (as IDs)
        rec_ids = self.recommender.recommend(
            item_ids, top_k=top_k, exclude_seen=exclude_seen
        )

        # Convert IDs back to names
        rec_names = self.mapper.ids_to_names(rec_ids)

        # Fill with top popular if needed
        if len(rec_names) < top_k:
            rec_names = self._fill_with_top_popular(
                rec_names, top_k, exclude_ids=item_ids if exclude_seen else []
            )

        return rec_names[:top_k]

    def _get_top_popular_names(self, top_k: int, exclude_ids: List[int]) -> List[str]:
        """
        Get top popular items as names.

        Args:
            top_k: Number of items to return
            exclude_ids: Item IDs to exclude

        Returns:
            List of top popular ingredient names
        """
        exclude_set = set(exclude_ids)
        popular_ids = []

        for item_id in self.top_popular.recommendations:
            if item_id not in exclude_set:
                popular_ids.append(item_id)
                if len(popular_ids) >= top_k:
                    break

        return self.mapper.ids_to_names(popular_ids)

    def _fill_with_top_popular(
        self, current_recs: List[str], top_k: int, exclude_ids: List[int]
    ) -> List[str]:
        """
        Fill recommendations with top popular items to reach top_k.

        Args:
            current_recs: Current recommendations (names)
            top_k: Target number of recommendations
            exclude_ids: Item IDs to exclude

        Returns:
            List of recommendations filled to top_k
        """
        if len(current_recs) >= top_k:
            return current_recs

        # Get IDs of current recommendations to avoid duplicates
        current_ids = set()
        for name in current_recs:
            item_id = self.mapper.get_id(name)
            if item_id is not None:
                current_ids.add(item_id)

        # Add excluded IDs
        exclude_set = set(exclude_ids) | current_ids

        # Get additional items from top popular
        needed = top_k - len(current_recs)
        additional = []

        for item_id in self.top_popular.recommendations:
            if item_id not in exclude_set:
                additional.append(item_id)
                if len(additional) >= needed:
                    break

        # Convert to names and append
        additional_names = self.mapper.ids_to_names(additional)
        return current_recs + additional_names

    def get_recommendations_with_scores(
        self, user_activity: List[str], top_k: int = 10, exclude_seen: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Get ingredient recommendations with relevance scores.
        Uses top popular items as fallback (with score 0.0) if needed.

        Args:
            user_activity: List of ingredient names the user has
            top_k: Number of recommendations to return
            exclude_seen: Whether to exclude ingredients user already has

        Returns:
            List of tuples (ingredient_name, score) - exactly top_k items
        """
        # Convert names to IDs
        item_ids = self.mapper.names_to_ids(
            user_activity, skip_unknown=True, warn_unknown=False
        )

        # If no valid items, use top popular with score 0.0
        if not item_ids:
            popular_names = self._get_top_popular_names(top_k, exclude_ids=[])
            return [(name, 0.0) for name in popular_names]

        # Get recommendations with names and scores
        recs_with_scores = self.recommender.recommend_with_names(
            item_ids, top_k=top_k, exclude_seen=exclude_seen
        )

        # Extract names and scores
        results = [(name, score) for _, name, score in recs_with_scores]

        # Fill with top popular if needed
        if len(results) < top_k:
            # Get current recommendation names
            current_names = [name for name, _ in results]

            # Fill to top_k
            filled_names = self._fill_with_top_popular(
                current_names, top_k, exclude_ids=item_ids if exclude_seen else []
            )

            # Add new items with score 0.0
            for name in filled_names[len(current_names) :]:
                results.append((name, 0.0))

        return results[:top_k]

    def get_recommendations_detailed(
        self, user_activity: List[str], top_k: int = 10, exclude_seen: bool = True
    ) -> dict:
        """
        Get detailed recommendation information.
        Uses top popular items as fallback if needed.

        Args:
            user_activity: List of ingredient names the user has
            top_k: Number of recommendations to return
            exclude_seen: Whether to exclude ingredients user already has

        Returns:
            Dictionary with:
                - user_ingredients: List of valid ingredients from input
                - recommendations: List of recommended ingredient names (exactly top_k)
                - scores: List of relevance scores (exactly top_k)
                - top_recommendations: List of (name, score) tuples (exactly top_k)
        """
        # Convert names to IDs
        item_ids = self.mapper.names_to_ids(
            user_activity, skip_unknown=True, warn_unknown=False
        )

        # If no valid items, use top popular
        if not item_ids:
            popular_names = self._get_top_popular_names(top_k, exclude_ids=[])
            return {
                "user_ingredients": [],
                "recommendations": popular_names,
                "scores": [0.0] * len(popular_names),
                "top_recommendations": [(name, 0.0) for name in popular_names],
            }

        # Get user ingredient names (validated)
        user_ingredients = self.mapper.ids_to_names(item_ids)

        # Get recommendations with scores (with fallback)
        recs_with_scores = self.get_recommendations_with_scores(
            user_activity, top_k=top_k, exclude_seen=exclude_seen
        )

        # Extract components
        recommendations = [name for name, _ in recs_with_scores]
        scores = [score for _, score in recs_with_scores]
        top_recommendations = recs_with_scores

        return {
            "user_ingredients": user_ingredients,
            "recommendations": recommendations,
            "scores": scores,
            "top_recommendations": top_recommendations,
        }

    def batch_get_recommendations(
        self,
        user_activities: List[List[str]],
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> List[List[str]]:
        """
        Get recommendations for multiple users at once.

        Args:
            user_activities: List of user activity lists
            top_k: Number of recommendations per user
            exclude_seen: Whether to exclude seen ingredients

        Returns:
            List of recommendation lists (one per user)
        """
        results = []

        for user_activity in user_activities:
            recs = self.get_recommendations(user_activity, top_k, exclude_seen)
            results.append(recs)

        return results
