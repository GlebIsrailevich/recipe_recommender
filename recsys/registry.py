from typing import Dict
from .models.popularity import PopularityRecommender
from .models.ease import EASERecommender
# from .models.sasrec import SASRecRecommender
# from .models.als import ALSRecommender
from recsys.models.ease_popular.inference import EasePopularRecommender
from recsys.models.llm_recs import LLMRecommender

import logging
logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Dict] = {
    "ease_popular_v1": {
        "class": EasePopularRecommender,
        "path": "models/ease_popular",
    },
    "llm_recs_v1": {
        "class": LLMRecommender,
        "path": "models/llm_recs",
    },
}

DEFAULT_EXPERIMENT = {
    "name": "main_recs_ab",
    "variants": {
        "ease_popular": 0.5,
        "llm_recs": 0.5,
        # можно вывести 0.8 / 0.2 и т.п.
    },
}

from .ab_testing import choose_variant_for_user  # noqa


def get_recommender_for_user(user_id: int):
    variant = choose_variant_for_user(user_id, DEFAULT_EXPERIMENT)

    cfg = MODEL_REGISTRY[variant]
    cls = cfg["class"]

    logger.info(
        "get_recommender_for_user: user_id=%s, experiment=%s, variant=%s, model_class=%s, model_path=%s",
        user_id,
        DEFAULT_EXPERIMENT["name"],
        variant,
        cls.__name__,
        cfg.get("path"),
    )

    model = cls()
    model.load(cfg["path"])
    return model
