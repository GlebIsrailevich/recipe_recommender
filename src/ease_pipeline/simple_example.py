"""
Simple Example - Quick Start with EASE Recommendations
"""

from main_ease import EASERecommendationSystem
from top_popular import TopPopular  # noqa: F401 - needed for pickle


def simple_example():
    """Simplest possible usage example"""

    # Step 1: Initialize the system (do this once)
    system = EASERecommendationSystem()

    # Step 2: Define user activity (ingredients they have)
    # user_activity = ["Молоко", "Яйцо куриное", "Масло сливочное"]
    # user_activity = ["Молоко", "Клубника", "Вишня", "Корица", "Крыжовник", "Сироп"]
    user_activity = ["Бекон", "Сыр", "Паста", "Соль", "Насвай"]

    # Step 3: Get recommendations
    recommendations = system.get_recommendations(user_activity, top_k=10)

    # Step 4: Display results
    print("\nUser has:")
    for ingredient in user_activity:
        print(f"  - {ingredient}")

    print("\nWe recommend adding:")
    for i, ingredient in enumerate(recommendations, 1):
        print(f"  {i}. {ingredient}")


def example_with_scores():
    """Example with relevance scores"""

    system = EASERecommendationSystem()

    user_activity = ["Вода", "Какао", "Сахар"]
    # user_activity = ["Чеснок", "Помидор", "Лук репчатый"]

    # Get recommendations with scores
    recommendations = system.get_recommendations_with_scores(user_activity, top_k=10)

    print("\nBased on:", user_activity)
    print("\nRecommendations (with relevance scores):")
    for i, (ingredient, score) in enumerate(recommendations, 1):
        print(f"  {i:2d}. {ingredient:30s} - {score * 100:5.1f}% match")


def batch_example():
    """Example with multiple users"""

    system = EASERecommendationSystem()

    # Multiple users with different ingredient sets
    # users = [
    #     ['Молоко', 'Яйцо куриное'],
    #     ['Чеснок', 'Помидор'],
    #     ['Мука пшеничная', 'Сахар-песок']
    # ]
    users = [
        ["Молоко", "Клубника", "Вишня", "Корица"],
        ["Вода", "Какао", "Сахар"],
        ["Кефир", "Грудка куриная", "Лимон"],
    ]

    # Get recommendations for all users at once
    all_recommendations = system.batch_get_recommendations(users, top_k=5)

    # Display results
    for i, (user_activity, recs) in enumerate(zip(users, all_recommendations), 1):
        print(f"\nUser {i}:")
        print(f"  Has: {', '.join(user_activity)}")
        print(f"  Recommendations: {', '.join(recs[:3])}, ...")


if __name__ == "__main__":
    print("=" * 70)
    print("EASE Recommendations - Simple Examples")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("Example 1: Basic Usage")
    print("=" * 70)
    simple_example()

    print("\n" + "=" * 70)
    print("Example 2: With Scores")
    print("=" * 70)
    example_with_scores()

    print("\n" + "=" * 70)
    print("Example 3: Batch Processing")
    print("=" * 70)
    batch_example()

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
