"""
Input Mapper - Convert ingredient names to item IDs and vice versa
Provides utilities for mapping between ingredient names and model IDs
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union


class IngredientMapper:
    """
    Maps between ingredient names and item IDs.

    Usage:
        mapper = IngredientMapper('recsys_tests/items_dict.json')

        # Convert names to IDs
        ids = mapper.names_to_ids(['Молоко', 'Яйцо куриное'])

        # Convert IDs to names
        names = mapper.ids_to_names([0, 4, 15])

        # Check if ingredient exists
        if mapper.has_ingredient('Молоко'):
            id = mapper.get_id('Молоко')
    """

    def __init__(self, names_path: str):
        """
        Initialize the mapper with ingredient name mappings.

        Args:
            names_path: Path to JSON file with id->name mappings
        """
        self.names_path = Path(names_path)

        # Load mappings
        with open(self.names_path, "r", encoding="utf-8") as f:
            self.id2name = json.load(f)

        # Convert string keys to int
        self.id2name: Dict[int, str] = {int(k): v for k, v in self.id2name.items()}

        # Create reverse mapping: name -> id
        self.name2id: Dict[str, int] = {v: k for k, v in self.id2name.items()}

        print(
            f"Loaded {len(self.id2name)} ingredient mappings from {self.names_path.name}"
        )

    def names_to_ids(
        self, names: List[str], skip_unknown: bool = True, warn_unknown: bool = True
    ) -> List[int]:
        """
        Convert ingredient names to item IDs.

        Args:
            names: List of ingredient names
            skip_unknown: If True, skip unknown ingredients; if False, raise error
            warn_unknown: If True, print warnings for unknown ingredients

        Returns:
            List of item IDs

        Raises:
            ValueError: If skip_unknown=False and unknown ingredient found
        """
        item_ids = []

        for name in names:
            if name in self.name2id:
                item_ids.append(self.name2id[name])
            else:
                if warn_unknown:
                    print(f"Warning: Ingredient '{name}' not found in mappings")
                if not skip_unknown:
                    raise ValueError(f"Unknown ingredient: '{name}'")

        return item_ids

    def ids_to_names(
        self, ids: List[int], default_format: str = "Unknown_{id}"
    ) -> List[str]:
        """
        Convert item IDs to ingredient names.

        Args:
            ids: List of item IDs
            default_format: Format string for unknown IDs (use {id} placeholder)

        Returns:
            List of ingredient names
        """
        names = []

        for item_id in ids:
            if item_id in self.id2name:
                names.append(self.id2name[item_id])
            else:
                names.append(default_format.format(id=item_id))

        return names

    def get_id(self, name: str) -> Optional[int]:
        """
        Get item ID for a single ingredient name.

        Args:
            name: Ingredient name

        Returns:
            Item ID or None if not found
        """
        return self.name2id.get(name)

    def get_name(self, item_id: int) -> Optional[str]:
        """
        Get ingredient name for a single item ID.

        Args:
            item_id: Item ID

        Returns:
            Ingredient name or None if not found
        """
        return self.id2name.get(item_id)

    def has_ingredient(self, name: str) -> bool:
        """Check if ingredient name exists in mappings."""
        return name in self.name2id

    def has_id(self, item_id: int) -> bool:
        """Check if item ID exists in mappings."""
        return item_id in self.id2name

    def convert_mixed(
        self,
        items: List[Union[int, str]],
        skip_unknown: bool = True,
        warn_unknown: bool = True,
    ) -> List[int]:
        """
        Convert mixed list of IDs and names to all IDs.

        Args:
            items: List containing item IDs (int) or names (str)
            skip_unknown: If True, skip unknown items
            warn_unknown: If True, print warnings for unknown items

        Returns:
            List of item IDs
        """
        item_ids = []

        for item in items:
            if isinstance(item, str):
                # It's a name, convert to ID
                if item in self.name2id:
                    item_ids.append(self.name2id[item])
                else:
                    if warn_unknown:
                        print(f"Warning: Ingredient '{item}' not found")
                    if not skip_unknown:
                        raise ValueError(f"Unknown ingredient: '{item}'")
            elif isinstance(item, (int, int)):
                # Already an ID
                if item in self.id2name or not warn_unknown:
                    item_ids.append(item)
                else:
                    if self.has_id(item):
                        item_ids.append(item)
                    else:
                        print(f"Warning: Item ID {item} not found in mappings")
                        if not skip_unknown:
                            raise ValueError(f"Unknown item ID: {item}")
            else:
                print(f"Warning: Unsupported item type: {type(item)}")

        return item_ids

    def get_all_ingredients(self) -> List[str]:
        """Get list of all available ingredient names."""
        return sorted(self.name2id.keys())

    def get_all_ids(self) -> List[int]:
        """Get list of all available item IDs."""
        return sorted(self.id2name.keys())

    def search_ingredients(self, query: str, limit: int = 10) -> List[tuple]:
        """
        Search for ingredients by partial name match.

        Args:
            query: Search query (case-insensitive)
            limit: Maximum number of results

        Returns:
            List of tuples (item_id, name) matching the query
        """
        query_lower = query.lower()
        results = []

        for item_id, name in self.id2name.items():
            if query_lower in name.lower():
                results.append((item_id, name))
                if len(results) >= limit:
                    break

        return results


def load_mapper(
    names_path: str,
) -> IngredientMapper:
    """
    Convenience function to load the ingredient mapper.

    Args:
        names_path: Path to the items dictionary JSON file

    Returns:
        Initialized IngredientMapper
    """
    return IngredientMapper(names_path)
