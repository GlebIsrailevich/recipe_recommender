import json
from typing import List


class InternalIDs2Names:
    def __init__(self):
        with open(
            "D:/gleb/Recsys_projeccts/sirius_seminars/vkusvill_case/src/matching_pipeline/vv_id2names.json",
            "r",
            encoding="utf-8",
        ) as f:
            id_to_name = json.load(f)

        self.internal2vkusvill = {int(k): v for k, v in id_to_name.items()}

    def ids_to_names(self, cart_items: List[int]) -> List[str]:
        vkussvil_names = []

        for id in cart_items:
            if id in self.internal2vkusvill:
                vkussvil_names.append(self.internal2vkusvill[id])
            else:
                print(f"Warning: Internal ID '{id}' not found in mappings")
        return vkussvil_names
