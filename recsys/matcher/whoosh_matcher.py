from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from whoosh import scoring
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, TEXT, Schema
from whoosh.filedb.filestore import RamStorage
from whoosh.query import And, FuzzyTerm, Or, Term


@dataclass(frozen=True)
class MatchResult:
    pov_id: int
    vv_id: int
    ingredient: str
    queries: List[str]


class WhooshCatalogMatcher:
    """Lightweight in-memory Whoosh matcher for arbitrary catalogues."""

    def __init__(self, df: pd.DataFrame, id_col: str = "id", text_col: str = "name"):
        normalized = (
            df[[id_col, text_col]]
            .dropna()
            .drop_duplicates(subset=[id_col])
            .rename(columns={id_col: "id", text_col: "name"})
        )
        normalized["id"] = normalized["id"].astype(int)
        normalized["name"] = normalized["name"].astype(str)
        self.df = normalized.reset_index(drop=True)
        self.schema = (
            Schema(  # схема для индекса (как именно будут храниться данные в индексе)
                id=ID(stored=True, unique=True),
                name=TEXT(stored=True, analyzer=StemmingAnalyzer()),
                name_exact=TEXT(stored=True),
            )
        )
        self._index = self._build_index()

    def _build_index(self):  # создаёт индекс для поиска
        storage = RamStorage()
        idx = storage.create_index(self.schema)
        writer = idx.writer()
        for row in self.df.itertuples():
            writer.add_document(id=str(row.id), name=row.name, name_exact=row.name)
        writer.commit()
        return idx

    def search(
        self, query: str, limit: int = 5
    ) -> pd.DataFrame:  # поиск по запросу (запрос - это название ингредиента)
        query = (query or "").strip()
        if not query:
            return self.df.head(0)

        terms = [token for token in query.lower().split() if token]
        if not terms:
            return self.df.head(0)

        subqueries = []
        # создаём подзапросы для каждого токена (каждого слова в запросе)
        for token in terms:
            if len(token) < 3:
                subqueries.append(Or([Term("name", token), Term("name_exact", token)]))
            else:
                subqueries.append(
                    Or(
                        [
                            FuzzyTerm("name", token, maxdist=1, prefixlength=2),
                            Term("name_exact", token),
                        ]
                    )
                )

        whoosh_query = And(subqueries)  # это общий запрос по всем подзапросам
        rows: List[Dict[str, str]] = []
        # создаём список для хранения результатов
        with self._index.searcher(weighting=scoring.BM25F()) as searcher:
            hits = searcher.search(whoosh_query, limit=limit)
            for hit in hits:
                rows.append({"id": int(hit["id"]), "name": hit["name"]})

        if not rows:
            return self.df.head(0)
        return pd.DataFrame(rows)
