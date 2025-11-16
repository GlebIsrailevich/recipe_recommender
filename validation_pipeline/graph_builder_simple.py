from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import networkx as nx
#преобразовано для работы с нашим пайплайном валидации

class SimpleBipartiteGraph:
    
    def __init__(self, df: pd.DataFrame, min_ingredient_freq: int = 3):
        self.df = df.copy()
        self.min_ingredient_freq = min_ingredient_freq
        self.graph = nx.Graph()
        self.recipe_nodes = set()
        self.ingredient_nodes = set()
        self.recipe_to_ingredients = {}
        self.ingredient_to_recipes = defaultdict(set)
    
    def prepare_data(self):
        """Подготовка данных - используем ингредиенты как есть"""
        # Извлекаем ингредиенты из normalized формата
        def extract_ingredients(row):
            ingredients = row.get('ingredients_normalized', {})
            if isinstance(ingredients, str):
                import ast
                try:
                    ingredients = ast.literal_eval(ingredients)
                except:
                    ingredients = {}
            if isinstance(ingredients, dict):
                return list(ingredients.keys())
            return []
        
        self.df['ingredients_list'] = self.df.apply(extract_ingredients, axis=1)
        self.df = self.df[self.df['ingredients_list'].apply(len) >= 3].reset_index(drop=True)
        
        # Подсчет частот
        ingredient_counts = defaultdict(int)
        for ingredients in tqdm(self.df['ingredients_list'], desc="Подсчет частот"):
            for ing in ingredients:
                ingredient_counts[ing] += 1
        
        frequent_ingredients = {
            ing for ing, count in ingredient_counts.items()
            if count >= self.min_ingredient_freq
        }
        
        print(f"   Всего уникальных ингредиентов: {len(ingredient_counts)}")
        print(f"   Частых ингредиентов (>={self.min_ingredient_freq}): {len(frequent_ingredients)}")
        
        self.df['ingredients_filtered'] = self.df['ingredients_list'].apply(
            lambda x: [ing for ing in x if ing in frequent_ingredients]
        )
        
        self.df = self.df[self.df['ingredients_filtered'].apply(len) >= 3].reset_index(drop=True)
        return self
    
    def build_graph(self):
        """Построение двудольного графа"""
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Добавление рецептов"):
            recipe_id = f"recipe_{idx}"
            ingredients = row['ingredients_filtered']
            
            self.graph.add_node(recipe_id, bipartite=0, name=row.get('name', f'recipe_{idx}'))
            self.recipe_nodes.add(recipe_id)
            
            self.recipe_to_ingredients[recipe_id] = ingredients
            
            for ingredient in ingredients:
                ing_id = f"ing_{ingredient}"
                
                if ing_id not in self.graph:
                    self.graph.add_node(ing_id, bipartite=1, name=ingredient)
                    self.ingredient_nodes.add(ing_id)
                
                if self.graph.has_edge(recipe_id, ing_id):
                    self.graph[recipe_id][ing_id]['weight'] += 1
                else:
                    self.graph.add_edge(recipe_id, ing_id, weight=1.0)
                
                self.ingredient_to_recipes[ing_id].add(recipe_id)
        
        print(f"Узлов рецептов: {len(self.recipe_nodes)}")
        print(f"Узлов ингредиентов: {len(self.ingredient_nodes)}")
        print(f"Ребер: {self.graph.number_of_edges()}")
        
        return self
    
    def compute_ingredient_weights(self):
        """Вычисление весов ингредиентов (TF-IDF like)"""
        total_recipes = len(self.recipe_nodes)
        
        for ing_id in tqdm(self.ingredient_nodes, desc="Вычисление весов"):
            recipes_with_ing = len(self.ingredient_to_recipes[ing_id])
            if recipes_with_ing == 0:
                continue
            idf = np.log(total_recipes / recipes_with_ing)
            
            for recipe_id in self.ingredient_to_recipes[ing_id]:
                edge_data = self.graph[recipe_id][ing_id]
                tf = edge_data['weight']
                edge_data['tfidf_weight'] = tf * idf
        
        return self


class SimpleGraphRecommender:
    """Упрощенный рекомендер с методом weighted_scoring_v2"""
    
    def __init__(self, graph_builder: SimpleBipartiteGraph):
        self.graph_builder = graph_builder
        self.graph = graph_builder.graph
        self.recipe_nodes = graph_builder.recipe_nodes
        self.ingredient_nodes = graph_builder.ingredient_nodes
        
        # Вычисляем популярность ингредиентов
        from collections import Counter
        self.ingredient_popularity = Counter()
        for recipe_id in self.recipe_nodes:
            ingredients = self.graph_builder.recipe_to_ingredients[recipe_id]
            self.ingredient_popularity.update(ingredients)
        
        print(f"\n SimpleGraphRecommender инициализирован")
        print(f"   Уникальных ингредиентов: {len(self.ingredient_popularity)}")
        print(f"   Топ-5 популярных: {list(self.ingredient_popularity.most_common(5))}")
    
    def recommend_ingredients(
        self,
        cart_ingredients: List[str],
        top_k: int = 10,
        method: str = 'weighted_scoring_v2',
    ) -> List[tuple]:
        """Рекомендация ингредиентов на основе корзины"""
        cart_ing_ids = [f"ing_{ing}" for ing in cart_ingredients if f"ing_{ing}" in self.ingredient_nodes]
        
        if not cart_ing_ids:
            return []
        
        if method == 'weighted_scoring_v2':
            return self._weighted_scoring_v2(cart_ing_ids, top_k)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _weighted_scoring_v2(
        self,
        cart_ing_ids: List[str],
        top_k: int,
        min_overlap_ratio: float = 0.3,
        diversity_penalty: float = 0.5,
        use_tfidf: bool = True,
    ) -> List[tuple]:
        """Weighted scoring v2 с diversity penalty"""
        recipe_scores = defaultdict(float)
        total_recipes = len(self.recipe_nodes)
        
        for ing_id in cart_ing_ids:
            connected_recipes = self.graph_builder.ingredient_to_recipes.get(ing_id, set())
            
            for recipe_id in connected_recipes:
                recipe_ingredients = self.graph_builder.recipe_to_ingredients[recipe_id]
                recipe_ing_ids = [f"ing_{i}" for i in recipe_ingredients]
                
                overlap = len(set(cart_ing_ids) & set(recipe_ing_ids))
                overlap_ratio = overlap / len(cart_ing_ids) if cart_ing_ids else 0
                
                if overlap_ratio < min_overlap_ratio:
                    continue
                
                completion_score = (overlap / len(recipe_ingredients)) ** 2
                
                if overlap == len(cart_ing_ids):
                    completion_score *= 2.0
                
                recipe_scores[recipe_id] += completion_score
        
        if recipe_scores:
            max_score = max(recipe_scores.values())
            recipe_scores = {r: s / max_score for r, s in recipe_scores.items()}
        
        ingredient_scores = defaultdict(float)
        
        for recipe_id, recipe_weight in recipe_scores.items():
            recipe_ingredients = self.graph_builder.recipe_to_ingredients[recipe_id]
            
            for ingredient in recipe_ingredients:
                ing_id = f"ing_{ingredient}"
                
                if ing_id in cart_ing_ids:
                    continue
                
                base_score = recipe_weight
                
                if use_tfidf:
                    if self.graph.has_edge(recipe_id, ing_id):
                        edge_data = self.graph[recipe_id][ing_id]
                        if 'tfidf_weight' in edge_data:
                            tfidf = edge_data['tfidf_weight']
                            base_score *= (1 + tfidf)
                
                # Diversity penalty
                popularity = self.ingredient_popularity.get(ingredient, 0)
                popularity_normalized = popularity / total_recipes if total_recipes > 0 else 0
                diversity_factor = 1.0 / (1.0 + diversity_penalty * popularity_normalized)
                
                final_score = base_score * diversity_factor
                ingredient_scores[ing_id] += final_score
        
        sorted_ingredients = sorted(
            ingredient_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k * 5]
        
        # Преобразуем обратно в названия
        result = []
        for ing_id, score in sorted_ingredients:
            ingredient_name = self.graph.nodes[ing_id]['name']
            result.append((ingredient_name, score))
            if len(result) >= top_k:
                break
        
        return result
