from __future__ import annotations

from collections.abc import Sequence
from typing import Any

#преобразовано для работы с нашим пайплайном валидации

def create_graph_predictor(
    recommender: Any,
    id2item: dict[int, str],
    top_k: int = 20,
    method: str = 'weighted_scoring_v2',
) -> Any:
    # Строим маппинги между item_id и названиями ингредиентов из графа
    # Получаем все ингредиенты из графа
    graph_ingredients = set()
    for recipe_id in recommender.recipe_nodes:
        ingredients = recommender.graph_builder.recipe_to_ingredients[recipe_id]
        graph_ingredients.update(ingredients)
    
    #item_id -> название из графа
    id_to_graph_name = {}
    graph_name_to_id = {}
    
    for item_id, item_name in id2item.items():
        # Пытаемся найти точное совпадение
        if item_name in graph_ingredients:
            id_to_graph_name[item_id] = item_name
            graph_name_to_id[item_name] = item_id
        else:
            # Для упрощенной версии просто пропускаем, если нет точного совпадения
            # (данные уже нормализованы в валидаторе)
            pass
    
    def predictor(item_ids: Sequence[int]) -> list[int]:
        """
        Преобразует item_id в названия, получает рекомендации из графа,
        возвращает item_id рекомендованных ингредиентов.
        """
        # Преобразуем item_id в названия ингредиентов (из графа)
        ingredient_names = []
        for item_id in item_ids:
            graph_name = id_to_graph_name.get(item_id)
            if graph_name:
                ingredient_names.append(graph_name)
        
        if not ingredient_names:
            return []
        
        # Получаем рекомендации из графовой модели
        recommendations = recommender.recommend_ingredients(
            ingredient_names,
            top_k=top_k * 3,  # Берем больше, т.к. некоторые могут не найтись в маппинге
            method=method,
        )
        
        # Преобразуем названия обратно в item_id
        recommended_item_ids = []
        seen_item_ids = set(item_ids)  # Исключаем уже имеющиеся
        
        for ingredient_name, score in recommendations:
            item_id = graph_name_to_id.get(ingredient_name)
            if item_id is not None and item_id not in seen_item_ids:
                recommended_item_ids.append(item_id)
                seen_item_ids.add(item_id)
                if len(recommended_item_ids) >= top_k:
                    break
        
        return recommended_item_ids
    return predictor