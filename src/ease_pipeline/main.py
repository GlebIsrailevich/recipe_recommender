from main_ease import EASERecommendationSystem
from top_popular import TopPopular  # noqa: F401 - needed for pickle

system = EASERecommendationSystem()


user_activity = ...  # Список List["str"]

user_activity = ["Бекон", "Сыр", "Паста", "Соль", "Насвай"]
recommendations = system.get_recommendations(user_activity, top_k=10)
print(recommendations)
