import os
from typing import List

from whoosh import scoring
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, TEXT, Schema
from whoosh.index import create_in, exists_in, open_dir
from whoosh.query import FuzzyTerm, Or, Term

# Global cache for the index
_ingredient_index = None


class LLMOutputSearch:
    def __init__(self, df):
        self.df = df

    def build_ingredient_index(self):
        """
        Builds and caches Whoosh index from ingredient dataframe.
        Call once at startup or when data changes.
        """
        global _ingredient_index

        if _ingredient_index is not None:
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

        if exists_in(index_dir):
            _ingredient_index = open_dir(index_dir)
        else:
            _ingredient_index = create_in(index_dir, schema)

            # Write all ingredients to index
            writer = _ingredient_index.writer()
            for _, row in self.df.iterrows():
                writer.add_document(
                    id=str(row["id"]), name=row["name"], name_exact=row["name"]
                )
            writer.commit()

        return _ingredient_index

    def search_ingredients_fuzzy(self, query: str, limit: int = 20):
        """
        Optimized BM25F fuzzy search for ingredients.

        Args:
            df: Your ingredient dataframe
            query: Search query (e.g., "молоко")
            limit: Maximum results to return

        Returns:
            List of matching rows from dataframe, sorted by relevance
        """
        query = (query or "").strip()
        if not query:
            return self.df.head(0)  # empty dataframe

        # Build/get cached index
        idx = self.build_ingredient_index(self.df)

        # Split query into terms
        terms = [w for w in query.lower().split() if w.strip()]
        if not terms:
            return self.df.head(0)

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
        from whoosh.query import And

        whoosh_query = And(subqueries)

        # Search with BM25F scoring
        result_ids = []
        with idx.searcher(weighting=scoring.BM25F()) as searcher:
            results = searcher.search(whoosh_query, limit=limit)
            for hit in results:
                result_ids.append(int(hit["id"]))

        if not result_ids:
            return self.df.head(0)  # empty dataframe

        # Return matching rows in relevance order
        result_df = self.df[self.df["id"].isin(result_ids)]

        # Preserve search relevance order
        id_to_position = {id_val: pos for pos, id_val in enumerate(result_ids)}
        result_df = result_df.copy()
        result_df["_relevance_order"] = result_df["id"].map(id_to_position)
        result_df = result_df.sort_values("_relevance_order").drop(
            "_relevance_order", axis=1
        )

        return result_df

    def search2list(self, llm_outputs: List[str]) -> List[int]:
        ids_list = []
        for output in llm_outputs:
            df = self.search_ingredients_fuzzy(output, limit=1)

            if df.empty:  # ← This is the correct emptiness check
                continue

            value = df["id"].iloc[0]  # safer than .item()
            # value = df["name"].iloc[0]   # safer than .item()

            ids_list.append(int(value))
            # ids_list.append(value)

        return ids_list
