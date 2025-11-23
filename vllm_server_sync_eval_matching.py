import sys
import time
import ast
import argparse
from typing import List, Dict

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

try:
    from .llm_matcher_optimized import LLMMatcher
except ImportError:
    from llm_matcher_optimized import LLMMatcher

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
- Generate exactly 10 search queries
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

EXTRA_ITEMS_PLAN = [0, 2, 4, 6, 8]


def _build_batch_test_carts() -> List[Dict[str, List[str]]]:
    templates = [
        {
            "name": "Панкейки на завтрак",
            "items": [
                "Молоко ультрапастеризованное 3,2%",
                "Яйца куриные",
                "Мука пшеничная Макфа",
                "Сахар-песок",
                "Сливочное масло Домик в деревне",
            ],
            "extras": [
                "Разрыхлитель теста Dr. Bakers",
                "Ванилин Dr.Bakers карамельный",
                "Кленовый сироп",
                "Черника мини",
                "Бананы",
                "Орехи кедровые очищенные",
                "Шоколадные капли",
                "Масса творожная с шоколадом 23%",
                "Соль Экстра пл/",
                "Кофе жареный молотый",
                "Лимон",
                "Сметана Ростагроэкорт 10%",
            ],
        },
        {
            "name": "Паста болоньезе",
            "items": [
                "Макаронные изделия Barilla",
                "Фарш из говядины",
                "Томаты в собственном соку",
                "Лук репчатый фермерский",
                "Чеснок",
                "Масло оливковое Extra Virgin, Турция",
                "Томатная паста",
            ],
            "extras": [
                "Морковь мытая",
                "Стебель сельдерея",
                "Красное сухое вино Жан Поль ",
                "Сливочное масло Домик в деревне",
                "Базилик красный",
                "Сыр Пармезан",
                "Сахар-песок",
                "Соус Чипотле",
                "Лавровый лист",
                "Свежемолотый перец",
                "Панчетта Паста",
                "Сливки Ростагроэкорт 10%",
            ],
        },
        {
            "name": "Суши сет",
            "items": [
                "Рис для суши Японка",
                "Лосось филе с кожей на гриле",
                "Чипсы нори Чим-Чим Сэндвич",
                "Соус соевый Kikkoman",
                "Уксус рисовый пищевой",
                "Авокадо крупное",
            ],
            "extras": [
                "Огурец свежий",
                "Икра леща сушено-вяленая",
                "Имбирь Чим-Чим маринованный",
                "Сыр Белая Фета",
                "Сыр творожный Сливочный",
                "Соус Терияки сладкий",
                "Креветки Agama Королевская ",
                "Смесь семян подсолнечника",
                "Лук зеленый",
                "Лайм",
                "Манго",
                "Масло кунжутное",
            ],
        },
        {
            "name": "Борщ классический",
            "items": [
                "Свекла",
                "Капуста белокочанная",
                "Говядина для бульона без кости",
                "Картофель",
                "Морковь",
                "Лук репчатый",
                "Томатная паста Помидорка",
            ],
            "extras": [
                "Чеснок Фермерский",
                "Лавровый лист",
                "Черный перец горошком",
                "Сало",
                "Фасоль стручковая",
                "Уксус столовый",
                "Сахар-песок",
                "Сметана Ростагроэкорт 10%",
                "Зелень укропа",
                "Хрен Дядя Ваня столовый с/б",
                "Чесночные гренки",
                "Майоран сушеный измельченный",
            ],
        },
        {
            "name": "Утренний кофе-тост",
            "items": ["Хлеб тостовый", "Кофе зерновой"],
            "extras": [
                "Масло сливочное ЭкоНива 82,5%",
                "Джем клубничный",
                "Молоко Parmalat Comfort ",
                "Сыр творожный Violette",
                "Авокадо",
                "Мед натуральный цветочный",
                "Арахисовая паста",
                "Корица",
                "Бананы",
                "Орехи грецкие",
            ],
        },
        {
            "name": "Смузи-перекус",
            "items": ["Замороженные ягоды"],
            "extras": [
                "Банан",
                "Мёд",
                "Йогурт натуральный",
                "Овсяные хлопья",
                "Семена чиа",
                "Арахисовая паста классическая",
                "Напиток миндальное молоко",
                "Какао-порошок",
                "Салат Шпинат",
                "Манго Кент",
                "Имбирь",
            ],
        },
        {
            "name": "Протеиновый батончик",
            "items": ["Каша протеиновая Овсяная", "Финики свежие"],
            "extras": [
                "Арахис обжаренный с нори",
                "Кешью жареный",
                "Какао-порошок",
                "Стружка кокосовая",
                "Мед Липовый",
                "Хлопья овсяные без глютена",
                "Соль Славяна помол №1 пакет",
                "Сахар Dr.Bakers Ванильный",
                "Изюм узбекский Семушка",
                "Лен коричневый",
            ],
        },
        {
            "name": "Салат-минутка",
            "items": ["Салат Романо", "Соус Слобода Цезарь 60%"],
            "extras": [
                "Филе куриное малое ГП",
                "Гренки ржано-пшеничные",
                "Сыр Пармезан",
                "Томаты черри красные на ветке",
                "Авокадо крупное",
                "Бекон свиной с/к, нарезка",
                "Лук красный",
                "Огурцы длинноплодные ",
                "Кедровые орехи Семушка",
                "Лимоны Абхазия",
            ],
        },
        {
            "name": "Барбекю уикенд",
            "items": [
                "Стейки свиные",
                "Колбаски гриль",
                "Крылышки куриные",
                "Картофель молодой",
                "Кукуруза в початках",
                "Соус барбекю",
                "Соус чесночный",
                "Булочки для бургеров",
                "Салат айсберг",
                "Помидоры черри",
                "Красный лук",
                "Маринованные огурчики",
                "Сыр чеддер",
                "Бекон",
                "Листья салата",
                "Соус ранч",
                "Чипсы",
                "Арбуз",
                "Лимонад",
                "Ананас",
                "Пряности для гриля",
                "Соус терияки",
            ],
            "extras": [
                "Соус сальса томатный",
                "Огурец свежий длинноплодный",
                "Перец халапеньо маринованный",
                "Маслины без косточек",
                "Соус чили сладкий",
                "Фольга для гриля",
                "Уголь древесный берёзовый",
                "Шампуры металлические",
                "Булочки бриошь для бургеров",
                "Набор одноразовых тарелок",
                "Салфетки бумажные",
                "Соус барбекю медовый",
            ],
        },
        {
            "name": "Азия wok",
            "items": [
                "Лапша пшеничная Удон",
                "Филе цыпленка-бройлера",
                "Морковь мытая",
                "Перец болгарский",
                "Соус соевый Sen Soy Классический пл/б",
            ],
            "extras": [
                "Имбирь",
                "Шампиньоны",
                "Капуста брокколи Мираторг ",
                "Масло кунжутное Real Tang ст/б Китай",
                "Чеснок сушеный молотый",
                "Капуста пекинская",
                "Ананасы консервированные",
                "Кешью сушеный Семушка",
                "Соус устричный",
                "Лук зеленый",
                "Соус терияки",
                "Гарнир Националь Булгур кунжут",
            ],
        },
    ]

    carts: List[Dict[str, List[str]]] = []
    for template in templates:
        base_items = template["items"] 
        extra_items = template["extras"]
        for variant_idx, extra_count in enumerate(EXTRA_ITEMS_PLAN, start=1):
            items = list(base_items)
            if extra_count:
                items.extend(extra_items[:extra_count])
            items = items[:20]
            carts.append(
                {
                    "name": f"{template['name']} v{variant_idx}",
                    "items": items,
                }
            )

    return carts


BATCH_TEST_CARTS = _build_batch_test_carts()


def initialize_model(model_name: str):
    """
    Инициализирует движок VLLM и связанный с ним токенизатор в синхронном режиме.
    Возвращает кортеж (engine, tokenizer).
    """
    print(f"Загрузка модели из '{model_name}'...")
    engine_args = EngineArgs(
        model=model_name,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        trust_remote_code=True
    )
    engine = LLMEngine.from_engine_args(engine_args)
    # Важно: получаем токенизатор именно из движка, чтобы гарантировать соответствие
    tokenizer = engine.tokenizer

    print("Модель и токенизатор успешно загружены.")
    return engine, tokenizer


def get_current_cart(user_id: int) -> list[str]:
    """
    Функция-заглушка, возвращающая текущую корзину пользователя.
    """
    prompt_str = input("Введите товары через '; ': ")
    prompt_list = [item.strip() for item in prompt_str.split(";") if item.strip()]

    return prompt_list

def get_previous_n_cart(user_id: int, n: int) -> list[str]:
    """Функция-заглушка, возвращающая прошлые n покупок."""
    return []

def get_simillar_n_cart(user_id: int, n: int, current_cart: list = None) -> list[str]:
    """Функция-заглушка, возвращающая n наиболее похожих прошлых покупок."""
    return []


def get_recs(
    engine: LLMEngine,
    tokenizer,
    user_id: int,
    context_window: int,
    cart_override: List[str] | None = None,
    prev_cart_override: List[str] | None = None,
    sim_cart_override: List[str] | None = None,
) -> str:
    """
    Синхронно генерирует рекомендации, используя LLMEngine.
    """
    curr_cart = cart_override if cart_override is not None else get_current_cart(user_id)
    prev_cart = (
        prev_cart_override
        if prev_cart_override is not None
        else get_previous_n_cart(user_id, context_window)
    )
    sim_cart = (
        sim_cart_override
        if sim_cart_override is not None
        else get_simillar_n_cart(user_id, context_window, curr_cart)
    )

    prev_cart_str = "; ".join(prev_cart)
    sim_cart_str = "; ".join(sim_cart)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            current_basket="; ".join(curr_cart),
            n=context_window,
            prev_cart=prev_cart_str,
            sim_cart=sim_cart_str
        )},
    ]

    final_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    sampling_params = SamplingParams(max_tokens=200, temperature=0.0)
    request_id = random_uuid()
    
    start_time = time.time()
    
    # ИСПРАВЛЕНИЕ: Передаем только prompt (строку), а не prompt_token_ids.
    # Движок сам выполнит токенизацию.
    engine.add_request(request_id, final_prompt, sampling_params)

    final_output = []
    while engine.has_unfinished_requests():
        # Выполняем шаг генерации
        request_outputs = engine.step()
        
        # Проверяем завершенные на этом шаге запросы
        for output in request_outputs:
            if output.finished:
                final_output.append(output)
    print(f"Время работы LLM: {time.time()-start_time:.2f} сек.")

    if not final_output or not final_output[0].outputs:
        return ""

    text = final_output[0].outputs[0].text
    return text.strip()


def parse_llm_response(text: str) -> list[str]:
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


def map_cart_items_to_pov_ids(cart_items: List[str], matcher: LLMMatcher) -> List[int]:
    """
    Конвертирует наименования товаров в корзине в pov_id через Whoosh-поиск.
    """
    pov_ids: List[int] = []
    seen: set[int] = set()

    for item in cart_items:
        if not item.strip():
            continue
        matches = matcher._map_llm_to_pov_ids([item], per_query_limit=1)
        if not matches:
            continue
        pov_id = matches[0][0]
        if pov_id not in seen:
            seen.add(pov_id)
            pov_ids.append(pov_id)
    return pov_ids


def diversify_with_trace(reranked: List[Dict], top_k: int) -> tuple[List[int], List[Dict]]:
    """
    Повторяет логику matcher._diversify_and_select, но возвращает
    также кандидатов и их pov_id, попавших в итоговый список. 
    """
    grouped: Dict[int, List[Dict]] = {} # словарь pov_id -> список кандидатов
    for candidate in reranked:
        grouped.setdefault(candidate["pov_id"], []).append(candidate.copy())

    result: List[int] = []
    selected: List[Dict] = []
    while grouped and len(result) < top_k:
        empty_keys = []
        for pov_id, items in list(grouped.items()):
            if not items:
                empty_keys.append(pov_id)
                continue

            item = items.pop(0)
            vv_id = item["vv_id"]

            if vv_id not in result:
                result.append(vv_id)
                selected.append(item)

            if len(result) == top_k:
                break

        for key in empty_keys:
            grouped.pop(key, None)

    return result, selected


def run_full_pipeline(
    matcher: LLMMatcher,
    cart_pov_ids: List[int],
    pov_matches: List[tuple[int, str, List[str]]],
    llm_outputs: List[str],
    top_k: int,
) -> tuple[List[int], set[str]]:
    """
    Прогоняет pov_matches через production-пайплайн matcher'a,
    возвращает финальные VV-id и множество LLM-запросов,
    которые дожили до переранжированного списка (до заполнения популярными).
    """
    if not pov_matches:
        return matcher.TOP_POPULAR_VV_IDS[:top_k], set()

    try:
        ease_scores = matcher._get_ease_scores(cart_pov_ids, pov_matches)
        vv_candidates = matcher._map_pov_to_vv_candidates(pov_matches)
        if not vv_candidates:
            return matcher.TOP_POPULAR_VV_IDS[:top_k], set()

        reranked = matcher._rerank_candidates(vv_candidates, ease_scores, llm_outputs)
        selected_vv_ids, selected_candidates = diversify_with_trace(reranked, top_k)

        llm_queries_after = set()
        for candidate in selected_candidates:
            for query in candidate.get("llm_queries", []):
                cleaned = query.strip().lower()
                if cleaned:
                    llm_queries_after.add(cleaned)

        final_vv_ids = matcher._fill_with_popular(selected_vv_ids, top_k)
        return final_vv_ids[:top_k], llm_queries_after
    except Exception as exc:
        print(f"Ошибка пайплайна matcher: {exc}")
        return matcher.TOP_POPULAR_VV_IDS[:top_k], set()


def evaluate_search_quality(
    engine: LLMEngine,
    tokenizer,
    matcher: LLMMatcher,
    context_window: int,
    top_k: int,
):
    """
    Прогоняет заранее собранные корзины через LLM + поиск
    и оценивает, какой процент запросов LLM не удалось сопоставить.
    """

    stats = []
    for idx, cart in enumerate(BATCH_TEST_CARTS, start=1):
        cart_items = cart.get("items", [])
        cart_pov_ids = map_cart_items_to_pov_ids(cart_items, matcher)
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(BATCH_TEST_CARTS)}] Корзина: {cart.get('name', 'Без названия')}")
        print("Содержимое:", "; ".join(cart_items))

        raw_result = get_recs(
            engine,
            tokenizer,
            user_id=idx,
            context_window=context_window,
            cart_override=cart_items,
            prev_cart_override=cart.get("prev"),
            sim_cart_override=cart.get("similar"),
        )

        llm_outputs = parse_llm_response(raw_result) if raw_result else []
        llm_outputs = [item.strip() for item in llm_outputs if item.strip()]

        if not llm_outputs:
            print("LLM не сгенерировала валидные запросы — пропускаем корзину.")
            continue

        pov_matches = matcher._map_llm_to_pov_ids(llm_outputs, per_query_limit=3) # для каждого ответа LLM возвращается до трёх POV‑ингредиентов (id, название и исходный запрос)
        matched_queries = { # 
            query.strip().lower()
            for _, _, queries in pov_matches
            for query in queries # список исходных запросов LLM для данного ингредиента
            if query.strip()
        }

        total = len(llm_outputs)
        matched = sum(1 for q in llm_outputs if q.lower() in matched_queries)
        unmatched_queries = [q for q in llm_outputs if q.lower() not in matched_queries]
        drop_pct = (total - matched) / total if total else 0.0

        final_ids, llm_queries_after = run_full_pipeline(
            matcher,
            cart_pov_ids,
            pov_matches,
            llm_outputs,
            top_k,
        )
        post_matched = sum(1 for q in llm_outputs if q.lower() in llm_queries_after)
        post_unmatched_queries = [q for q in llm_outputs if q.lower() not in llm_queries_after]
        post_drop_pct = (total - post_matched) / total if total else 0.0

        print(f"LLM -> {total} запросов: {llm_outputs}")
        print(f"Сопоставлено через поиск: {matched}/{total} "
              f"({(matched/total)*100:.1f}%); отсеяно: {drop_pct*100:.1f}%")
        if unmatched_queries:
            print("Несопоставленные запросы:", ", ".join(unmatched_queries))
        print(f"После полного пайплайна: {post_matched}/{total} "
              f"({(post_matched/total)*100:.1f}% проходит, "
              f"{post_drop_pct*100:.1f}% отсеяно)")
        if post_unmatched_queries:
            print("Отсеяны после реранжирования:", ", ".join(post_unmatched_queries))
        print(f"Итоговые VV-id (top-{top_k}): {final_ids}")

        stats.append(
            {
                "name": cart.get("name", f"cart_{idx}"),
                "total": total,
                "search_matched": matched,
                "search_drop": drop_pct,
                "final_matched": post_matched,
                "final_drop": post_drop_pct,
            }
        )

    if stats:
        avg_search_drop = sum(item["search_drop"] for item in stats) / len(stats)
        avg_final_drop = sum(item["final_drop"] for item in stats) / len(stats)
        print("\n" + "=" * 80)
        print("Сводка по корзинам:")
        for item in stats:
            print(
                f"- {item['name']}: поиск {item['search_matched']}/{item['total']} "
                f"({(1 - item['search_drop'])*100:.1f}% проходит, "
                f"{item['search_drop']*100:.1f}% отсеяно); "
                f"финал {item['final_matched']}/{item['total']} "
                f"({(1 - item['final_drop'])*100:.1f}% проходит, "
                f"{item['final_drop']*100:.1f}% отсеяно)"
            )
        print(f"\nСреднее отсечение на поиске: {avg_search_drop*100:.1f}% "
              f"({(1-avg_search_drop)*100:.1f}% проходит)")
        print(f"Среднее отсечение после пайплайна: {avg_final_drop*100:.1f}% "
              f"({(1-avg_final_drop)*100:.1f}% рекомендаций сохраняется)")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Синхронный сервер VLLM + оценка качества поиска"
    )
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen3-14b",
        help="Путь или алиас модели HuggingFace",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=5,
        help="Количество прошлых корзин, добавляемых в промпт",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Количество итоговых рекомендаций matcher'а",
    )
    parser.add_argument(
        "--batch-eval",
        action="store_true",
        help="Запустить цикл оценки качества поиска на подготовленных корзинах",
    )
    return parser


def main():
    """
    Основной синхронный цикл программы.
    """
    args = build_arg_parser().parse_args()
    engine, tokenizer = initialize_model(args.model_path)

    if args.batch_eval:
        matcher = LLMMatcher()
        evaluate_search_quality(
            engine,
            tokenizer,
            matcher,
            context_window=args.context_window,
            top_k=args.top_k,
        )
        return

    try:
        while True:
            raw_result = get_recs(engine, tokenizer, 1, args.context_window)
            if raw_result:
                parsed_result = parse_llm_response(raw_result)
                print("Сгенерированные рекомендации:", parsed_result)
            else:
                print("Не удалось сгенерировать рекомендации.")

    except (KeyboardInterrupt, SystemExit):
        print("\nПрограмма завершена.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    main()