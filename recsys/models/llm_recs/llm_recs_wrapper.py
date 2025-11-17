# recsys/models/llm_recs/llm_recs_wrapper.py
from typing import List, Optional, Dict, Any
from recsys.base import BaseRecommender
from .main import LLMWrapped

class LLMRecommender(BaseRecommender):
    def __init__(self):
        self.system: Optional[LLMWrapped] = None

    def load(self, model_path: str):
        if self.system is not None:
            return  # уже загружено

        self.system = LLMWrapped()

    def recommend(
        self,
        user_id: int,
        cart_items: List[int],
        k: int = 10,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        if self.system is None:
            self.load("")
        if not cart_items:
            return []
        return self.system.recommend(cart_items, user_id=user_id, k=k)
