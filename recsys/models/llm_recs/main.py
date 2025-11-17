# recsys/models/llm_recs/main.py
from .vllm_server_sync import vllm_recomender
from .internal_id_names import InternalIDs2Names
from .llm_output_mapper import LLMOutputSearchDB

class LLMWrapped:
    def __init__(self):
        self.int_ids = InternalIDs2Names()
        self.out_ids = LLMOutputSearchDB()
        self.model = vllm_recomender()

    def idx2idx_llm_recs(self, input_idxs, user_id, k=10):
        names = self.int_ids.ids_to_names(input_idxs)
        if not names:
            return []
        cart_str = "; ".join(names)
        llm_raw = self.model.get_recs(cart_str, user_id, k)
        return self.out_ids.search2list(llm_raw, limit=k)
