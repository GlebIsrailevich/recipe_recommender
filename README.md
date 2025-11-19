# recipe_recommender

## ML Pipeline cross sell рекомендации FMCG продуктов
А теперь на понятном: система рекомендации новых продуктов Вкуссвилл в корзине на основе ранее добавленных 

## Основные стадии работы EASE pipeline

1. На сервисе юзер добавляет в корзину товары -> мы принимаем List[int] id товаров Вкуссвил

2. Маппим эти айдишники на наши внутренние айдишники (на которых мы учили модель EASE)

3. Передаем их в модель EASE

4. Получаем рекомендации модели EASE в виде внутренних айдишников

5. Маппим обратно внутренние айдишники на айдишники Вкуссвил через словарь -> отдаем юзеру рекомендации

[EASE Pipeline](https://github.com/GlebIsrailevich/recipe_recommender/tree/gleb_branch/src/ease_model_feat)


## Основные стадиии  работы LLM Pipeline

1. На сервисе юзер добавляет в корзину товары -> мы принимаем List[int] id товаров Вкуссвил [Переводим id в имена](https://github.com/GlebIsrailevich/recipe_recommender/blob/gleb_branch/src/llm_pipeline/internal_id_names.py)

2. Маппим эти айдишники на имена из Вкуссвила 

3. Передаем их в промпт LLM 

4. LLM отдает рекомендации в формате List[str] 

5. Рекомендации LLM мы матчим с ингридиентами поваренка с помощью поиска по Fuzzy и Term а затем BM25

6. Полученный айдишники мы маппим их на ids Вкуссвила

7. Передаем на реранкер где по скору EASE, цене и релевантости ранжируем

8. K рекомендаций отдаем юзеру

[LLM Matching Pipeline](https://github.com/GlebIsrailevich/recipe_recommender/tree/gleb_branch/src/matcher_llm_feat)

[Подготовка данных](https://github.com/GlebIsrailevich/recipe_recommender/tree/gleb_branch/src/preprocess_and_train)


## Fallback Case

Если рекомендации от моделей ломаются то мы делаем Fallback на TopPopular, если длина наших рекомендаций < k, то добиваем ToPopular. Сам TopPopular мы считаем по нашей бд, т.е. действия юзера влияют на результат [TopPopular](https://github.com/GlebIsrailevich/recipe_recommender/blob/gleb_branch/src/ease_model_feat/actual_popular.py)