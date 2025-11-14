def get_extra_model(model, user_interactions, seen_items, unseen, k=10):
    """
    Формирует рекомендательный список, комбинируя результаты модели и заранее заданные элементы.

    Параметры:
        model: обученная рекомендательная модель, у которой есть метод `recommend`.
        user_interactions: данные о взаимодействиях пользователя (формат зависит от модели).
        seen_items: множество или список объектов, которые пользователь уже видел.
        unseen: список заранее подготовленных рекомендаций, которые должны быть включены в результат.
        k (int): требуемое количество финальных рекомендаций.

    Возвращает:
        list: список из k рекомендаций, состоящий из unseen и дополненный менее релевантными
        рекомендациями модели, которые ещё не были просмотрены пользователем.

    """
    recommendations = model.recommend(user_interactions, top_k=k + 20)

    print(recommendations)
    recs_to_generate = k - len(unseen)

    less_relevant_model = [item for item in recommendations if item not in seen_items][
        :recs_to_generate
    ]

    new_recos = unseen + less_relevant_model
    # print(new_recos)
    assert len(new_recos) == k

    return new_recos


def add_top_pop(seen_items, unseen, top_pop, k=10):
    """
    Формирует список рекомендаций, добавляя популярные элементы, которых пользователь ещё не видел.

    Параметры:
        seen_items: элементы, которые пользователь уже смотрел.
        unseen: список элементов, которые обязательно должны попасть в выдачу.
        top_pop: список популярных элементов, из которых добираются рекомендации.
        k (int): итоговое количество рекомендаций.

    Возвращает:
        list: список из k рекомендаций, объединяющий unseen и элементы из топ-популярных,
        которые не встречаются в seen_items.


    """

    recs_to_generate = k - len(unseen)

    top_pop_recs = [item for item in top_pop if item not in seen_items][
        :recs_to_generate
    ]

    new_recos = unseen + top_pop_recs
    assert len(new_recos) == k

    return new_recos
