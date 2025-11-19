import os
from typing import List

from sqlalchemy import select
from sqlalchemy.orm import Session
from whoosh import scoring
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, TEXT, Schema
from whoosh.index import create_in, exists_in, open_dir
from whoosh.query import And, FuzzyTerm, Or, Term

# Global cache for the index
_ingredient_index = None


class LLMOutputSearchDB:
    def __init__(self, db_session: Session, model_class):
        """
        Initialize search with database session and model.

        Args:
            db_session: SQLAlchemy session
            model_class: Your SQLAlchemy model class (e.g., Ingredient)
                        Should have 'id' and 'name' columns
        """
        self.db_session = db_session
        self.model_class = model_class

    def build_ingredient_index(self, force_rebuild: bool = False):
        """
        Builds and caches Whoosh index from database.
        Call once at startup or when data changes.

        Args:
            force_rebuild: If True, rebuilds index even if it exists
        """
        global _ingredient_index

        if _ingredient_index is not None and not force_rebuild:
            return _ingredient_index

        # Define schema
        schema = Schema(
            id=ID(stored=True, unique=True),
            name=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            name_exact=TEXT(stored=True),  # for exact matching
        )

        # Create index in memory (faster) or on disk
        index_dir = "ingredient_catalog_index"
        if not os.path.exists(index_dir):
            os.mkdir(index_dir)

        if exists_in(index_dir) and not force_rebuild:
            _ingredient_index = open_dir(index_dir)
        else:
            _ingredient_index = create_in(index_dir, schema)

            # Write all ingredients to index from database
            writer = _ingredient_index.writer()

            # Query all records from database
            stmt = select(self.model_class)
            results = self.db_session.execute(stmt).scalars().all()

            for row in results:
                writer.add_document(id=str(row.id), name=row.name, name_exact=row.name)
            writer.commit()

        return _ingredient_index

    def search_ingredients_fuzzy(self, query: str, limit: int = 20):
        """
        Optimized BM25F fuzzy search for ingredients.

        Args:
            query: Search query (e.g., "молоко")
            limit: Maximum results to return

        Returns:
            List of model instances from database, sorted by relevance
        """
        query = (query or "").strip()
        if not query:
            return []

        # Build/get cached index
        idx = self.build_ingredient_index()

        # Split query into terms
        terms = [w for w in query.lower().split() if w.strip()]
        if not terms:
            return []

        # Build query for each term
        subqueries = []
        for t in terms:
            if len(t) < 3:
                # Short terms: exact match only (no fuzzy to avoid noise)
                subqueries.append(
                    Or(
                        [
                            Term("name", t),
                            Term("name_exact", t),
                        ]
                    )
                )
            else:
                # Longer terms: fuzzy matching with typo tolerance
                # maxdist=1 means 1 character difference allowed
                # prefixlength=2 means first 2 chars must match exactly (performance optimization)
                subqueries.append(
                    Or(
                        [
                            FuzzyTerm("name", t, maxdist=1, prefixlength=2),
                            Term("name_exact", t),
                        ]
                    )
                )

        # Combine all terms (all must match)
        whoosh_query = And(subqueries)

        # Search with BM25F scoring
        result_ids = []
        with idx.searcher(weighting=scoring.BM25F()) as searcher:
            results = searcher.search(whoosh_query, limit=limit)
            for hit in results:
                result_ids.append(int(hit["id"]))

        if not result_ids:
            return []

        # Fetch matching rows from database in relevance order
        # Create a mapping to preserve search order
        id_to_position = {id_val: pos for pos, id_val in enumerate(result_ids)}

        # Query database for matching IDs
        stmt = select(self.model_class).where(self.model_class.id.in_(result_ids))
        db_results = self.db_session.execute(stmt).scalars().all()

        # Sort by relevance order
        sorted_results = sorted(db_results, key=lambda x: id_to_position.get(x.id, 999))

        return sorted_results

    def search2list(self, llm_outputs: List[str]) -> List[int]:
        """
        Search for multiple LLM outputs and return list of IDs.

        Args:
            llm_outputs: List of search queries

        Returns:
            List of ingredient IDs
        """
        ids_list = []
        for output in llm_outputs:
            results = self.search_ingredients_fuzzy(output, limit=1)

            if not results:  # Empty list check
                continue

            # Get the first result's ID
            ids_list.append(results[0].id)

        return ids_list


# Example usage:
"""
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Ingredient(Base):
    __tablename__ = 'ingredients'
    
    id = Column(Integer, primary_key=True)
    name = Column(String)

# Create engine and session
engine = create_engine('sqlite:///ingredients.db')
Session = sessionmaker(bind=engine)
session = Session()

# Initialize search
searcher = LLMOutputSearchDB(session, Ingredient)

# Build index (do this once at startup)
searcher.build_ingredient_index()

# Search
results = searcher.search_ingredients_fuzzy("молоко", limit=5)
for ingredient in results:
    print(f"{ingredient.id}: {ingredient.name}")

# Search multiple outputs
llm_outputs = ["молоко", "мука", "яйца"]
ids = searcher.search2list(llm_outputs)
print(ids)  # [123, 456, 789]
"""
