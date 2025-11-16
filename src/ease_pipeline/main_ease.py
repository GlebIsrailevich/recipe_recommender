"""
Main EASE Recommender Interface
Simple API for getting ingredient recommendations from user activity
"""

import pickle
from typing import List, Tuple

from config import EASEMODEL, MAPPING, POV2VV, TOPPOPULAR, VV2POV
from ease_map import EASERecommenderWithNames
from matcher2id import VkussvillMapper
from top_popular import TopPopular  # noqa: F401 - needed for pickle


class EASERecommendationSystem:
    """
    Main interface for EASE recommendations.
    Takes Vkusvill product IDs as input and returns Vkusvill product IDs as output.

    Usage:
        # Initialize the system
        system = EASERecommendationSystem()

        # Get recommendations from user activity (Vkusvill IDs)
        user_activity = [4862, 2964, 8964, 1040]
        recommendations = system.get_recommendations(user_activity, top_k=10)

        print(f"Based on: {user_activity}")
        print(f"We recommend: {recommendations}")
    """

    def __init__(
        self,
        model_path: str = EASEMODEL,
        names_path: str = MAPPING,
        toppop_path=TOPPOPULAR,
        pov2vv: str = POV2VV,
        vv2pov: str = VV2POV,
    ):
        """
        Initialize the recommendation system.

        Args:
            model_path: Path to the EASE model pickle file
            names_path: Path to the ingredient names JSON file
            toppop_path: Path to the top popular model pickle file
            vkusvill_mapping_path: Path to the local-to-vkusvill ID mapping JSON file
        """
        print("Initializing EASE Recommendation System...")
        print("-" * 70)

        # Load ingredient mapper (for local IDs to names)
        # self.mapper = IngredientMapper(names_path)

        # Load Vkusvill mapper (for internal IDs to external Vkusvill IDs)
        self.vkusvill_mapper = VkussvillMapper(vv2pov, pov2vv)

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
        self, user_activity: List[int], top_k: int = 10, exclude_seen: bool = True
    ) -> List[int]:
        """
        Get product recommendations based on user activity.
        Uses top popular items as fallback if EASE returns insufficient recommendations.

        Args:
            user_activity: List of Vkusvill product IDs the user has
            top_k: Number of recommendations to return
            exclude_seen: Whether to exclude products user already has

        Returns:
            List of recommended Vkusvill product IDs (exactly top_k items)
        """
        # Convert external Vkusvill IDs to internal IDs
        internal_ids = self.vkusvill_mapper.vkusvill_to_internal(user_activity)

        # If no valid items, use top popular
        if not internal_ids:
            return self._get_top_popular_vkusvill_ids(top_k, exclude_internal_ids=[])

        # Get recommendations (as internal IDs)
        rec_internal_ids = self.recommender.recommend(
            internal_ids, top_k=top_k, exclude_seen=exclude_seen
        )

        # Convert internal IDs to external Vkusvill IDs
        rec_vkusvill_ids = self.vkusvill_mapper.internal_to_vkusvill(
            rec_internal_ids, skip_unknown=True, warn_unknown=False
        )

        # Fill with top popular if needed
        if len(rec_vkusvill_ids) < top_k:
            rec_vkusvill_ids = self._fill_with_top_popular(
                rec_vkusvill_ids,
                top_k,
                exclude_internal_ids=internal_ids if exclude_seen else [],
            )

        return rec_vkusvill_ids[:top_k]

    def _get_top_popular_vkusvill_ids(
        self, top_k: int, exclude_internal_ids: List[int]
    ) -> List[int]:
        """
        Get top popular items as Vkusvill IDs.

        Args:
            top_k: Number of items to return
            exclude_internal_ids: Internal item IDs to exclude

        Returns:
            List of top popular Vkusvill product IDs
        """
        exclude_set = set(exclude_internal_ids)
        popular_internal_ids = []

        for internal_id in self.top_popular.recommendations:
            if internal_id not in exclude_set:
                popular_internal_ids.append(internal_id)
                if len(popular_internal_ids) >= top_k:
                    break

        return self.vkusvill_mapper.internal_to_vkusvill(
            popular_internal_ids, skip_unknown=True, warn_unknown=False
        )

    def _fill_with_top_popular(
        self, current_recs: List[int], top_k: int, exclude_internal_ids: List[int]
    ) -> List[int]:
        """
        Fill recommendations with top popular items to reach top_k.

        Args:
            current_recs: Current recommendations (Vkusvill IDs)
            top_k: Target number of recommendations
            exclude_internal_ids: Internal item IDs to exclude

        Returns:
            List of recommendations filled to top_k
        """
        if len(current_recs) >= top_k:
            return current_recs

        # Get internal IDs of current recommendations to avoid duplicates
        current_internal_ids = set()
        for vkusvill_id in current_recs:
            internal_id = self.vkusvill_mapper.get_internal_id(vkusvill_id)
            if internal_id is not None:
                current_internal_ids.add(internal_id)

        # Add excluded IDs
        exclude_set = set(exclude_internal_ids) | current_internal_ids

        # Get additional items from top popular
        needed = top_k - len(current_recs)
        additional_internal_ids = []

        for internal_id in self.top_popular.recommendations:
            if internal_id not in exclude_set:
                additional_internal_ids.append(internal_id)
                if len(additional_internal_ids) >= needed:
                    break

        # Convert to Vkusvill IDs and append
        additional_vkusvill_ids = self.vkusvill_mapper.internal_to_vkusvill(
            additional_internal_ids, skip_unknown=True, warn_unknown=False
        )
        return current_recs + additional_vkusvill_ids

    def get_recommendations_with_scores(
        self, user_activity: List[int], top_k: int = 10, exclude_seen: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Get product recommendations with relevance scores.
        Uses top popular items as fallback (with score 0.0) if needed.

        Args:
            user_activity: List of Vkusvill product IDs the user has
            top_k: Number of recommendations to return
            exclude_seen: Whether to exclude products user already has

        Returns:
            List of tuples (vkusvill_id, score) - exactly top_k items
        """
        # Convert external Vkusvill IDs to internal IDs
        internal_ids = self.vkusvill_mapper.vkusvill_to_internal(user_activity)

        # If no valid items, use top popular with score 0.0
        if not internal_ids:
            popular_vkusvill_ids = self._get_top_popular_vkusvill_ids(
                top_k, exclude_internal_ids=[]
            )
            return [(vk_id, 0.0) for vk_id in popular_vkusvill_ids]

        # Get recommendations with internal IDs and scores
        recs_with_scores = self.recommender.recommend_with_names(
            internal_ids, top_k=top_k, exclude_seen=exclude_seen
        )

        # Extract internal IDs and scores, convert to Vkusvill IDs
        results = []
        for internal_id, _, score in recs_with_scores:
            vkusvill_id = self.vkusvill_mapper.get_vkusvill_id(internal_id)
            if vkusvill_id is not None:
                results.append((vkusvill_id, score))

        # Fill with top popular if needed
        if len(results) < top_k:
            # Get current recommendation Vkusvill IDs
            current_vkusvill_ids = [vk_id for vk_id, _ in results]

            # Fill to top_k
            filled_vkusvill_ids = self._fill_with_top_popular(
                current_vkusvill_ids,
                top_k,
                exclude_internal_ids=internal_ids if exclude_seen else [],
            )

            # Add new items with score 0.0
            for vk_id in filled_vkusvill_ids[len(current_vkusvill_ids) :]:
                results.append((vk_id, 0.0))

        return results[:top_k]

    def get_recommendations_detailed(
        self, user_activity: List[int], top_k: int = 10, exclude_seen: bool = True
    ) -> dict:
        """
        Get detailed recommendation information.
        Uses top popular items as fallback if needed.

        Args:
            user_activity: List of Vkusvill product IDs the user has
            top_k: Number of recommendations to return
            exclude_seen: Whether to exclude products user already has

        Returns:
            Dictionary with:
                - user_products: List of valid Vkusvill IDs from input
                - recommendations: List of recommended Vkusvill IDs (exactly top_k)
                - scores: List of relevance scores (exactly top_k)
                - top_recommendations: List of (vkusvill_id, score) tuples (exactly top_k)
        """
        # Convert external Vkusvill IDs to internal IDs
        internal_ids = self.vkusvill_mapper.vkusvill_to_internal(user_activity)

        # If no valid items, use top popular
        if not internal_ids:
            popular_vkusvill_ids = self._get_top_popular_vkusvill_ids(
                top_k, exclude_internal_ids=[]
            )
            return {
                "user_products": [],
                "recommendations": popular_vkusvill_ids,
                "scores": [0.0] * len(popular_vkusvill_ids),
                "top_recommendations": [(vk_id, 0.0) for vk_id in popular_vkusvill_ids],
            }

        # Get user product IDs (validated - convert back to Vkusvill IDs)
        user_vkusvill_ids = self.vkusvill_mapper.internal_to_vkusvill(
            internal_ids, skip_unknown=True, warn_unknown=False
        )

        # Get recommendations with scores (with fallback)
        recs_with_scores = self.get_recommendations_with_scores(
            user_activity, top_k=top_k, exclude_seen=exclude_seen
        )

        # Extract components
        recommendations = [vk_id for vk_id, _ in recs_with_scores]
        scores = [score for _, score in recs_with_scores]
        top_recommendations = recs_with_scores

        return {
            "user_products": user_vkusvill_ids,
            "recommendations": recommendations,
            "scores": scores,
            "top_recommendations": top_recommendations,
        }

    def batch_get_recommendations(
        self,
        user_activities: List[List[int]],
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> List[List[int]]:
        """
        Get recommendations for multiple users at once.

        Args:
            user_activities: List of user activity lists (Vkusvill IDs)
            top_k: Number of recommendations per user
            exclude_seen: Whether to exclude seen products

        Returns:
            List of recommendation lists (one per user, Vkusvill IDs)
        """
        results = []

        for user_activity in user_activities:
            recs = self.get_recommendations(user_activity, top_k, exclude_seen)
            results.append(recs)

        return results
