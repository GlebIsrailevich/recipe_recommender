from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable
from vllm.engine.llm_engine import LLMEngine
from transformers import PreTrainedTokenizer

from validate import RecommendationValidator
from whoosh_matcher import WhooshIngredientMatcher, load_whoosh_matcher_from_validator

SYSTEM_PROMPT = """/no-think You are an AI assistant for a
grocery recommendation system.
Your task is to analyze a shopping cart and generate search queries
for missing ingredients needed to complete recipes.

IMPORTANT INSTRUCTIONS:
1. Analyze the current cart items and identify what recipes they might be used for
2. Consider the user's purchase history and similar users patterns when available
3. Generate search queries that help users find missing ingredients for cooking
4. Focus on general categories and cooking ideas, not specific branded products or weights
5. Ensure queries are realistic for a grocery store with 10,000-15,000 items
6. Avoid repeating items already in the cart
7. Avoid duplicates in your generated list

OUTPUT FORMAT:
- You MUST respond ONLY with a Python list format: ["query1", "query2", ...]
- All queries MUST be in Russian language
- Generate exactly 10 or more search queries
- Do not include any explanations, comments, or additional text
- The output should be a valid Python list that can be parsed with ast.literal_eval()

EXAMPLES:
Input cart: ["Мука пшеничная", "Яйца куриные", "Сахар-песок"]
Correct output: ["Разрыхлитель", "Молоко","Ванилин", "Сливочное масло", "Шоколад", "Кондитерские украшения", "Ванильный сахар", "Корица", "Какао", "Орехи"]
Incorrect output: ["Розрыхлитель", "Гедза", "Бумага"]

Remember: Think about what dishes can be made with the current ingredients, then suggest what's missing."""

USER_PROMPT_TEMPLATE = """CONTEXT:
Current user basket: {current_basket}
Previous {n} baskets' of user: {prev_cart}
Previous {n} mostly similar baskets' of user: {sim_cart}
YOUR OUTPUT: 
"""
#функции для работы с LLM
# Устаревшие функции - теперь используем WhooshMatcher, оставлены для обратной совместимости, но не используются
def find_ingredient_match(
    query: str,
    item2id: dict[str, int],
    threshold: float = 0.6,
    case_sensitive: bool = False,
    whoosh_matcher: WhooshIngredientMatcher | None = None
) -> int | None:
    """
    Находит наиболее похожий ингредиент используя Whoosh поиск
    
    Args:
        query: Поисковый запрос (строка от LLM)
        item2id: Словарь {название_ингредиента: item_id} (для обратной совместимости)
        threshold: Не используется (для обратной совместимости)
        case_sensitive: Не используется (для обратной совместимости)
        whoosh_matcher: WhooshIngredientMatcher для поиска
    
    Returns:
        item_id если найден подходящий ингредиент, иначе None
    """
    if not query or whoosh_matcher is None:
        return None
    
    return whoosh_matcher.find_match(query, limit=3)


def map_ingredient_names_to_ids(
    ingredient_names: list[str],
    item2id: dict[str, int],
    threshold: float = 0.6,
    verbose: bool = False,
    whoosh_matcher: WhooshIngredientMatcher | None = None
) -> list[int]:
    """
    Преобразует список названий ингредиентов (от LLM) в список item_id используя Whoosh
    
    Args:
        ingredient_names: Список названий ингредиентов от LLM
        item2id: Словарь {название_ингредиента: item_id} (для обратной совместимости)
        threshold: Не используется (для обратной совместимости)
        verbose: Выводить ли информацию о не найденных ингредиентах
        whoosh_matcher: WhooshIngredientMatcher для поиска
    
    Returns:
        Список item_id (только успешно сопоставленные)
    """
    if whoosh_matcher is None:
        return []
    
    return whoosh_matcher.map_names_to_ids(ingredient_names, verbose=verbose)


def get_llm_recommendations(
    engine: LLMEngine,
    tokenizer: PreTrainedTokenizer,
    current_basket: list[str],
    prev_cart: list[str] | None = None,
    sim_cart: list[str] | None = None,
    context_window: int = 5
) -> list[str]:
    """
    Получает рекомендации от LLM для заданной корзины.
    
    Args:
        engine: LLMEngine из vllm
        tokenizer: Токенизатор
        current_basket: Список названий ингредиентов в текущей корзине
        prev_cart: Список прошлых покупок (опционально)
        sim_cart: Список похожих покупок (опционально)
        context_window: Размер контекстного окна
    
    Returns:
        Список названий рекомендованных ингредиентов
    """
    import ast
    from vllm.sampling_params import SamplingParams
    from vllm.utils import random_uuid
    
    prev_cart = prev_cart or []
    sim_cart = sim_cart or []
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            current_basket="; ".join(current_basket),
            n=context_window,
            prev_cart="; ".join(prev_cart),
            sim_cart="; ".join(sim_cart)
        )},
    ]
    
    final_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    prompt_tokens = len(tokenizer.encode(final_prompt))
    
    sampling_params = SamplingParams(max_tokens=200, temperature=0.0)
    request_id = random_uuid()
    
    engine.add_request(request_id, final_prompt, sampling_params)
    
    final_output = []
    while engine.has_unfinished_requests():
        request_outputs = engine.step()
        for output in request_outputs:
            if output.finished:
                final_output.append(output)
    
    if not final_output or not final_output[0].outputs:
        return []
    
    text = final_output[0].outputs[0].text.strip()
    
    # Парсим ответ LLM
    try:
        start_index = text.find('[')
        end_index = text.rfind(']')
        
        if start_index == -1 or end_index == -1:
            return []
        
        parsed_list = ast.literal_eval(text[start_index : end_index + 1])
        
        if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
            return parsed_list
        else:
            return []
    except (ValueError, SyntaxError, TypeError):
        return []


def create_llm_predictor_with_user_context(
    engine: LLMEngine,
    tokenizer: PreTrainedTokenizer,
    validator: RecommendationValidator,
    context_window: int = 5,
    matching_threshold: float = 0.6,
    prev_cart_fn: Callable[[int, int], list[str]] | None = None,
    sim_cart_fn: Callable[[int, int, list[int]], list[str]] | None = None,
    verbose: bool = True,
    user_context_map: dict[int, dict[str, list[str]]] | None = None
) -> tuple[Callable[[Sequence[int], int], list[int]], Callable]:
    """
    Создает предиктор с поддержкой user_id для использования с RecommendationValidator
    Возвращает кортеж (предиктор_с_user_id, обертка_для_валидатора)
    
    Args:
        engine: LLMEngine из vllm
        tokenizer: Токенизатор
        validator: Экземпляр RecommendationValidator
        context_window: Размер контекстного окна для LLM
        matching_threshold: Порог схожести для маппинга названий в индексы (не используется с Whoosh)
        prev_cart_fn: Функция для получения прошлых покупок (user_id, n) -> list[str]
        sim_cart_fn: Функция для получения похожих покупок (user_id, n, current_items) -> list[str]
        verbose: Выводить ли дополнительную информацию
        user_context_map: Словарь {user_id: {'prev_cart': [...], 'sim_cart': [...]}} для предзагруженного контекста
    
    Returns:
        Кортеж (предиктор_с_user_id, обертка_для_валидатора)
    """
    id2item = validator.id2item
    item2id = validator.item2id
    user_context_map = user_context_map or {}
    
    # Создаем WhooshMatcher для поиска
    whoosh_matcher = load_whoosh_matcher_from_validator(validator)
    
    def llm_predictor_with_user(item_ids: Sequence[int], user_id: int) -> list[int]:
        """Предиктор, который принимает item_ids и user_id."""
        # Преобразуем индексы в названия
        current_basket = [id2item.get(int(item_id), '') for item_id in item_ids]
        current_basket = [name for name in current_basket if name]
        
        if not current_basket:
            return []
        
        prev_cart = []
        sim_cart = []
        
        if user_id in user_context_map:
            context = user_context_map[user_id]
            prev_cart = context.get('prev_cart', [])
            sim_cart = context.get('sim_cart', [])
        else:
            if prev_cart_fn:
                prev_cart = prev_cart_fn(user_id, context_window)
            if sim_cart_fn:
                sim_cart = sim_cart_fn(user_id, context_window, list(item_ids))
        
        # Получаем рекомендации от LLM
        llm_recommendations = get_llm_recommendations(
            engine=engine,
            tokenizer=tokenizer,
            current_basket=current_basket,
            prev_cart=prev_cart,
            sim_cart=sim_cart,
            context_window=context_window
        )
        
        if verbose and llm_recommendations:
            print(f"User {user_id}: LLM рекомендации: {llm_recommendations[:5]}...")
        
        # Преобразуем названия обратно в индексы используя Whoosh
        matched_ids = map_ingredient_names_to_ids(
            llm_recommendations,
            item2id,
            threshold=matching_threshold,
            verbose=verbose,
            whoosh_matcher=whoosh_matcher
        )
        
        # Исключаем уже имеющиеся в корзине ингредиенты
        item_ids_set = set(int(i) for i in item_ids)
        filtered_ids = [item_id for item_id in matched_ids if item_id not in item_ids_set]
        
        return filtered_ids
    
    # Обертка для валидатора
    def llm_predictor_wrapper(item_ids: Sequence[int], **kwargs) -> list[int]:
        user_id = kwargs.get('user_id', None)
        if user_id is not None:
            return llm_predictor_with_user(item_ids, user_id)
        return llm_predictor_with_user(item_ids, -1)
    
    return llm_predictor_with_user, llm_predictor_wrapper


def create_llm_predictor(
    engine: LLMEngine,
    tokenizer: PreTrainedTokenizer,
    validator: RecommendationValidator,
    context_window: int = 5,
    matching_threshold: float = 0.6,
    prev_cart_fn: Callable[[int, int], list[str]] | None = None,
    sim_cart_fn: Callable[[int, int, list[int]], list[str]] | None = None,
    verbose: bool = False,
    user_context_map: dict[int, dict[str, list[str]]] | None = None
) -> Callable[[Sequence[int]], list[int]]:
    """
    Создает предиктор для использования с RecommendationValidator.
    Это упрощенная версия, которая работает без user_id.
    
    Args:
        engine: LLMEngine из vllm
        tokenizer: Токенизатор
        validator: Экземпляр RecommendationValidator (для доступа к item2id, id2item)
        context_window: Размер контекстного окна для LLM
        matching_threshold: Порог схожести для маппинга названий в индексы
        prev_cart_fn: Функция для получения прошлых покупок (user_id, n) -> list[str]
        sim_cart_fn: Функция для получения похожих покупок (user_id, n, current_items) -> list[str]
        verbose: Выводить ли дополнительную информацию
        user_context_map: Словарь {user_id: {'prev_cart': [...], 'sim_cart': [...]}} для предзагруженного контекста
    
    Returns:
        Функция-предиктор, совместимая с RecommendationValidator
    """
    _, wrapper = create_llm_predictor_with_user_context(
        engine=engine,
        tokenizer=tokenizer,
        validator=validator,
        context_window=context_window,
        matching_threshold=matching_threshold,
        prev_cart_fn=prev_cart_fn,
        sim_cart_fn=sim_cart_fn,
        verbose=verbose,
        user_context_map=user_context_map
    )
    return wrapper


def create_llm_predictor_with_auto_user_id(
    engine: LLMEngine,
    tokenizer: PreTrainedTokenizer,
    validator: RecommendationValidator,
    split: str,
    context_window: int = 5,
    matching_threshold: float = 0.6,
    prev_cart_fn: Callable[[int, int], list[str]] | None = None,
    sim_cart_fn: Callable[[int, int, list[int]], list[str]] | None = None,
    verbose: bool = False,
    user_context_map: dict[int, dict[str, list[str]]] | None = None
) -> Callable[[Sequence[int]], list[int]]:
    """
    Создает предиктор с автоматическим получением user_id из контекста валидатора.
    Эта версия создает замыкание, которое знает о user_id для каждого запроса.
    
    Args:
        engine: LLMEngine из vllm
        tokenizer: Токенизатор
        validator: Экземпляр RecommendationValidator
        split: Сплит для которого создается предиктор ('train', 'val', 'test')
        context_window: Размер контекстного окна для LLM
        matching_threshold: Порог схожести для маппинга названий в индексы
        prev_cart_fn: Функция для получения прошлых покупок (user_id, n) -> list[str]
        sim_cart_fn: Функция для получения похожих покупок (user_id, n, current_items) -> list[str]
        verbose: Выводить ли дополнительную информацию
        user_context_map: Словарь {user_id: {'prev_cart': [...], 'sim_cart': [...]}} для предзагруженного контекста
    
    Returns:
        Функция-предиктор, совместимая с RecommendationValidator
    """
    # Получаем mapping item_ids -> user_id для данного сплита
    split_df = validator.get_grouped_split(split)
    item_to_user_map: dict[tuple[int, ...], int] = {}
    for _, row in split_df.iterrows():
        support_items = tuple(sorted(validator._ensure_list(row['support_items'])))
        item_to_user_map[support_items] = row['user_id']
    
    predictor_with_user, _ = create_llm_predictor_with_user_context(
        engine=engine,
        tokenizer=tokenizer,
        validator=validator,
        context_window=context_window,
        matching_threshold=matching_threshold,
        prev_cart_fn=prev_cart_fn,
        sim_cart_fn=sim_cart_fn,
        verbose=verbose,
        user_context_map=user_context_map
    )
    
    def llm_predictor_with_auto_user(item_ids: Sequence[int], **kwargs) -> list[int]:
        """
        Предиктор, который автоматически определяет user_id из контекста.
        """
        item_ids_tuple = tuple(sorted(int(i) for i in item_ids))
        user_id = item_to_user_map.get(item_ids_tuple, -1)
        
        if user_id == -1 and verbose:
            print(f"Warning: user_id не найден для items {item_ids[:5]}...")
        
        return predictor_with_user(item_ids, user_id)
    
    return llm_predictor_with_auto_user

