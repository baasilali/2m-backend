import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import process
from typing import Dict, List, Tuple, Any, Optional
import pickle

# Class to handle both embedding-based and fuzzy search
class SkinSearchEngine:
    def __init__(self, data_path: str = None, embedding_cache_path: str = None, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the search engine with both embedding-based and fuzzy search capabilities
        
        Args:
            data_path: Path to the JSON file with skin data
            embedding_cache_path: Path to save/load embeddings cache
            model_name: Name of the sentence-transformer model to use
        """
        # Safely load the model - handles both older and newer huggingface-hub versions
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error loading model with default settings: {e}")
            # Try with no_cache=True which works with newer huggingface-hub versions
            try:
                from sentence_transformers import SentenceTransformer
                print("Trying alternative model loading method for newer huggingface-hub versions...")
                self.model = SentenceTransformer(model_name, cache_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache"))
            except Exception as e2:
                print(f"Error loading model with alternative method: {e2}")
                raise RuntimeError(f"Failed to load SentenceTransformer model: {e} AND {e2}")
        
        self.index = None
        self.items = {}
        self.item_names = []
        self.embeddings = None
        self.embedding_cache_path = embedding_cache_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "data", 
            "embeddings_cache.pkl"
        )
        
        # Initialize data if path is provided
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path: str):
        """Load skin data from JSON file and prepare for search"""
        # Load the marketplace data JSON
        with open(data_path, 'r') as file:
            marketplace_data = json.load(file)
        
        # Handle the new skinport_data.json structure
        if isinstance(marketplace_data, list):
            # Convert list of items to dictionary with market_hash_name as key
            self.items = {item['market_hash_name']: item for item in marketplace_data}
        else:
            # Handle old format or other structures
            if "marketplace_data" in marketplace_data:
                self.items = marketplace_data.get("marketplace_data", {})
            else:
                self.items = marketplace_data
        
        # Create a list of item names for fuzzy matching
        self.item_names = list(self.items.keys())
        
        # Create or load embeddings
        self._create_or_load_embeddings()
    
    def _create_or_load_embeddings(self):
        """Create embeddings for all item names or load from cache if available"""
        # Check if cache exists
        if os.path.exists(self.embedding_cache_path):
            try:
                with open(self.embedding_cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Only use cache if the item names match exactly
                if sorted(cache_data['item_names']) == sorted(self.item_names):
                    self.embeddings = cache_data['embeddings']
                    # Create the FAISS index
                    self._create_faiss_index()
                    print(f"Loaded embeddings from cache for {len(self.item_names)} items")
                    return
            except Exception as e:
                print(f"Error loading embedding cache: {e}")
        
        # Create embeddings if not loaded from cache
        if self.item_names:
            # For each item, create searchable text that includes variations of the name
            search_texts = []
            for item_name in self.item_names:
                # Create variations to improve search results
                parts = item_name.split("|")
                base_name = parts[0].strip() if len(parts) > 1 else item_name
                skin_name = parts[1].strip() if len(parts) > 1 else ""
                
                # Additional metadata to include in search text
                wear = ""
                if "Factory New" in item_name:
                    wear = "Factory New FN"
                elif "Minimal Wear" in item_name:
                    wear = "Minimal Wear MW"
                elif "Field-Tested" in item_name:
                    wear = "Field-Tested FT"
                elif "Well-Worn" in item_name:
                    wear = "Well-Worn WW"
                elif "Battle-Scarred" in item_name:
                    wear = "Battle-Scarred BS"
                
                # Combine all information into a searchable text
                search_text = f"{item_name} {base_name} {skin_name} {wear}"
                search_texts.append(search_text)
            
            # Generate embeddings
            self.embeddings = self.model.encode(search_texts)
            
            # Create the FAISS index
            self._create_faiss_index()
            
            # Save embeddings to cache
            self._save_embeddings_cache()
            
            print(f"Created new embeddings for {len(self.item_names)} items")
    
    def _create_faiss_index(self):
        """Create a FAISS index for efficient similarity search"""
        if self.embeddings is not None and len(self.embeddings) > 0:
            # Initialize index - we use L2 norm which corresponds to Euclidean distance
            d = self.embeddings.shape[1]  # embedding dimension
            self.index = faiss.IndexFlatL2(d)
            # Convert embeddings to the correct format and add to index
            self.index.add(self.embeddings.astype(np.float32))
    
    def _save_embeddings_cache(self):
        """Save embeddings to cache file"""
        if self.embeddings is not None and self.item_names:
            cache_dir = os.path.dirname(self.embedding_cache_path)
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                
            with open(self.embedding_cache_path, 'wb') as f:
                pickle.dump({
                    'item_names': self.item_names,
                    'embeddings': self.embeddings
                }, f)
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Perform semantic search to find items similar to the query
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            List of tuples containing (item_name, similarity_score)
        """
        if not self.index or not self.item_names:
            return []
        
        # Encode the query
        query_embedding = self.model.encode([query])
        
        # Search in the FAISS index
        distances, indices = self.index.search(query_embedding.astype(np.float32), min(top_k, len(self.item_names)))
        
        # Extract results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.item_names):
                # Convert distance to similarity score (smaller distance = higher similarity)
                similarity = 1.0 / (1.0 + distances[0][i])
                results.append((self.item_names[idx], similarity))
        
        return results
    
    def fuzzy_search(self, query: str, top_k: int = 3) -> List[Tuple[str, int]]:
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
        
        # Use process.extract for fuzzy matching
        return process.extract(query, self.item_names, limit=top_k)
    
    def hybrid_search(self, query: str, top_k: int = 5, 
                      semantic_weight: float = 0.7, 
                      min_score_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Perform a hybrid search combining semantic search and fuzzy matching
        
        Args:
            query: The search query
            top_k: Number of top results to return
            semantic_weight: Weight to give semantic search (0-1)
            min_score_threshold: Minimum normalized score to include result
            
        Returns:
            List of dictionaries with item name and score information
        """
        # Perform semantic search
        semantic_results = self.semantic_search(query, top_k=top_k*2)
        
        # Perform fuzzy search
        fuzzy_results = self.fuzzy_search(query, top_k=top_k*2)
        
        # Combine and normalize results
        combined_results = {}
        
        # Add semantic results
        for item_name, score in semantic_results:
            combined_results[item_name] = {
                'item_name': item_name,
                'semantic_score': score,
                'fuzzy_score': 0,
                'total_score': semantic_weight * score
            }
        
        # Add fuzzy results (normalize scores to 0-1 range)
        max_fuzzy_score = 100  # fuzzywuzzy scores are between 0-100
        for item_name, score in fuzzy_results:
            normalized_score = score / max_fuzzy_score
            if item_name in combined_results:
                # Update existing entry
                combined_results[item_name]['fuzzy_score'] = normalized_score
                combined_results[item_name]['total_score'] += (1 - semantic_weight) * normalized_score
            else:
                # Create new entry
                combined_results[item_name] = {
                    'item_name': item_name,
                    'semantic_score': 0,
                    'fuzzy_score': normalized_score,
                    'total_score': (1 - semantic_weight) * normalized_score
                }
        
        # Sort by total score and filter by threshold
        results = list(combined_results.values())
        results = [r for r in results if r['total_score'] >= min_score_threshold]
        results.sort(key=lambda x: x['total_score'], reverse=True)
        
        return results[:top_k]
    
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
        
        # Check for specific weapon types
        for weapon in weapon_types:
            if weapon in query.lower():
                expanded_terms.append(weapon)
        
        # Check for wear conditions
        for wear in wear_types:
            if wear in query.lower():
                expanded_terms.append(wear)
        
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
        
        # Add expanded terms to query
        if expanded_terms:
            return f"{query} {' '.join(expanded_terms)}"
        
        return query


# Initialize a global instance to be imported by other modules
def get_skin_search_engine(data_path: Optional[str] = None) -> SkinSearchEngine:
    """Get or create the skin search engine singleton"""
    if not hasattr(get_skin_search_engine, 'instance') or get_skin_search_engine.instance is None:
        if data_path is None:
            # Use default path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(current_dir, "data", "prices_output.json")
        
        get_skin_search_engine.instance = SkinSearchEngine(data_path)
    
    return get_skin_search_engine.instance 