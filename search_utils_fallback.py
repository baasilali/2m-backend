import os
import json
from fuzzywuzzy import process
from typing import Dict, List, Tuple, Any, Optional

class SkinSearchEngineFallback:
    """
    A simpler version of the search engine that only uses fuzzy matching,
    without requiring sentence-transformers or FAISS.
    """
    def __init__(self, data_path: str = None):
        """
        Initialize the search engine with fuzzy search capabilities
        
        Args:
            data_path: Path to the JSON file with skin data
        """
        self.items = {}
        self.item_names = []
        
        # Initialize data if path is provided
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path: str):
        """Load skin data from JSON file and prepare for search"""
        # Load the marketplace data JSON
        with open(data_path, 'r') as file:
            marketplace_data = json.load(file)
        
        # Extract items
        if "marketplace_data" in marketplace_data:
            self.items = marketplace_data.get("marketplace_data", {})
        else:
            self.items = marketplace_data
        
        # Create a list of item names for fuzzy matching
        self.item_names = list(self.items.keys())
        
        print(f"Fallback search engine loaded with {len(self.item_names)} items")
    
    def fuzzy_search(self, query: str, top_k: int = 5) -> List[Tuple[str, int]]:
        """
        Perform fuzzy matching to find items with names similar to the query
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            List of tuples containing (item_name, match_score)
        """
        if not self.item_names:
            return []
        
        # Prepare the query by adding variations
        expanded_query = self._expand_query(query)
        
        # Use process.extract for fuzzy matching
        results = process.extract(expanded_query, self.item_names, limit=top_k)
        return results
    
    def _expand_query(self, query: str) -> str:
        """
        Expand a query with common skin terms to improve matching
        """
        # Check for common weapon types
        weapon_types = ["ak-47", "m4a4", "m4a1-s", "awp", "desert eagle", "deagle", 
                        "knife", "karambit", "bayonet", "butterfly", "gloves"]
        wear_types = ["factory new", "fn", "minimal wear", "mw", "field-tested", "ft", 
                     "well-worn", "ww", "battle-scarred", "bs"]
        
        # Add some common terms if not present
        expanded_terms = []
        
        # Check for gloves
        if "glove" in query.lower() or "gloves" in query.lower():
            if not any(term in query.lower() for term in ["sport gloves", "driver gloves", "specialist gloves", 
                                                         "moto gloves", "hand wraps", "bloodhound gloves"]):
                expanded_terms.append("sport gloves")
        
        # Check for specific patterns
        if "fade" in query.lower() and not any(w in query.lower() for w in weapon_types):
            expanded_terms.append("knife")
        
        if "doppler" in query.lower() and not any(w in query.lower() for w in weapon_types):
            expanded_terms.append("knife")
        
        # Check for wear if a skin is mentioned but no wear
        has_skin_term = any(term in query.lower() for term in ["fade", "doppler", "tiger", "marble", "slaughter", 
                                                              "crimson", "case hardened", "autotronic", "lore", 
                                                              "gamma", "emerald", "sapphire", "ruby", "rust", 
                                                              "damascus", "ultraviolet", "night", "blue steel", 
                                                              "stained", "forest", "boreal", "safari", "scorched", 
                                                              "urban masked"])
        if has_skin_term and not any(w in query.lower() for w in wear_types):
            expanded_terms.append("factory new")
        
        # Add the expanded terms if any
        if expanded_terms:
            expanded_query = f"{query} {' '.join(expanded_terms)}"
            return expanded_query
        
        return query
    
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Fallback method that just uses fuzzy search but follows the interface of the main search engine
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with item name and score information
        """
        # Perform fuzzy search
        fuzzy_results = self.fuzzy_search(query, top_k=top_k)
        
        # Format results to match the main search engine's output format
        results = []
        for item_name, score in fuzzy_results:
            normalized_score = score / 100.0  # fuzzywuzzy scores are 0-100
            results.append({
                'item_name': item_name,
                'semantic_score': 0,  # No semantic score in fallback
                'fuzzy_score': normalized_score,
                'total_score': normalized_score
            })
        
        return results
    
    def search(self, query: str, top_k: int = 5) -> List[str]:
        """
        Perform search and return just the item names in ranked order
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            List of item names ranked by relevance
        """
        results = self.hybrid_search(query, top_k=top_k)
        return [r['item_name'] for r in results]


# Initialize a global instance to be imported by other modules
def get_skin_search_engine(data_path: Optional[str] = None) -> SkinSearchEngineFallback:
    """Get or create the skin search engine singleton"""
    if not hasattr(get_skin_search_engine, 'instance') or get_skin_search_engine.instance is None:
        if data_path is None:
            # Use default path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(current_dir, "data", "prices_output.json")
        
        get_skin_search_engine.instance = SkinSearchEngineFallback(data_path)
    
    return get_skin_search_engine.instance 