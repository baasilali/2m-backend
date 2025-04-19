import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import process
from typing import Dict, List, Tuple, Any, Optional
import pickle
from pathlib import Path

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
        self.model_name = model_name
        self.model = None
        self.index = None
        self.skin_data = []
        self.items = {}
        self.item_names = []
        self.embeddings = None
        self.embedding_cache_path = embedding_cache_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "data", 
            "embeddings_cache.pkl"
        )
        
        # Load model first
        self.load_model()
        
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
        
        # Special handling for search terms that match the beginning of items
        # This helps prioritize exact prefix matches alongside semantic matches
        prefix_matches = []
        query_lower = query.lower()
        
        for item_name in self.item_names:
            item_lower = item_name.lower()
            
            # Check if item name starts with the query
            if item_lower.startswith(query_lower):
                # Perfect prefix match
                prefix_matches.append((item_name, 1.0))
            # Check if item contains the query as a word
            elif f" {query_lower} " in f" {item_lower} ":
                # Contains the exact query as a word
                prefix_matches.append((item_name, 0.9))
            # Check if any word in the item starts with the query
            elif any(word.startswith(query_lower) for word in item_lower.split()):
                # Word starts with query
                prefix_matches.append((item_name, 0.8))
                
        # Perform semantic search if we haven't found enough prefix matches
        if len(prefix_matches) < top_k:
            try:
                # Encode the query
                query_embedding = self.model.encode([query])
                
                # Search in the FAISS index
                distances, indices = self.index.search(query_embedding.astype(np.float32), min(top_k*2, len(self.item_names)))
                
                # Extract results
                semantic_results = []
                for i, idx in enumerate(indices[0]):
                    if idx < len(self.item_names):
                        # Convert distance to similarity score (smaller distance = higher similarity)
                        similarity = 1.0 / (1.0 + distances[0][i])
                        item_name = self.item_names[idx]
                        
                        # Only add if not already in prefix matches
                        if not any(item_name == pm[0] for pm in prefix_matches):
                            semantic_results.append((item_name, similarity))
                
                # Combine results (prefix matches first, then semantic)
                combined_results = prefix_matches + semantic_results
                combined_results.sort(key=lambda x: x[1], reverse=True)
                
                return combined_results[:top_k]
            except Exception as e:
                print(f"Error during semantic search: {e}")
                # If semantic search fails, return whatever prefix matches we found
                return prefix_matches[:top_k]
        else:
            # If we have enough prefix matches, just return those
            return prefix_matches[:top_k]
    
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
            
        # Try to extract keywords that might be part of skin names
        query_parts = query.lower().split()
        
        # Use regular fuzzy matching first
        main_results = process.extract(query, self.item_names, limit=top_k)
        results_dict = {name: score for name, score in main_results}
        
        # Also try matching individual parts if there are multiple words
        if len(query_parts) > 1:
            for part in query_parts:
                # Skip very short parts and common weapon names
                if len(part) < 3 or part in ["ak", "m4", "awp", "usp", "p90"]:
                    continue
                    
                # Look for items that contain this part
                for item_name in self.item_names:
                    if part in item_name.lower() and item_name not in results_dict:
                        # Add this result with a moderate score
                        results_dict[item_name] = 70  # Base score for partial matches
        
        # Convert back to list format
        results = [(name, score) for name, score in results_dict.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
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
        # Expand the query first (which now includes normalization)
        expanded_query = self._expand_query(query)
        
        # Also try the original query in case the normalization missed something
        results_expanded = {}
        results_original = {}
        
        # Try expanded/normalized query
        semantic_results_expanded = self.semantic_search(expanded_query, top_k=top_k*3)
        fuzzy_results_expanded = self.fuzzy_search(expanded_query, top_k=top_k*3)
        
        # Try original query as well if it's different
        if expanded_query != query:
            semantic_results_original = self.semantic_search(query, top_k=top_k*2)
            fuzzy_results_original = self.fuzzy_search(query, top_k=top_k*2)
            
            # Process original query results
            for item_name, score in semantic_results_original:
                results_original[item_name] = score * 0.7
                
            for item_name, score in fuzzy_results_original:
                if item_name in results_original:
                    results_original[item_name] += (score / 100.0) * 0.3
                else:
                    results_original[item_name] = (score / 100.0) * 0.3
        
        # Process expanded query results
        for item_name, score in semantic_results_expanded:
            results_expanded[item_name] = score * 0.7
            
        for item_name, score in fuzzy_results_expanded:
            if item_name in results_expanded:
                results_expanded[item_name] += (score / 100.0) * 0.3
            else:
                results_expanded[item_name] = (score / 100.0) * 0.3
        
        # Combine results, favoring normalized/expanded query but including unique results from original
        combined_results = results_expanded.copy()
        
        for item_name, score in results_original.items():
            if item_name in combined_results:
                # Slightly boost score if it appeared in both queries
                combined_results[item_name] = max(combined_results[item_name], score) * 1.1
            else:
                combined_results[item_name] = score
        
        # Print debug info
        normalized_query_info = f"Original: '{query}', Normalized: '{expanded_query}'"
        if query != expanded_query:
            print(f"Query normalized: {normalized_query_info}")
                
        # Sort by combined score
        sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return [item_name for item_name, _ in sorted_results[:top_k]]

    def _expand_query(self, query: str) -> str:
        """
        Expand and normalize a query with common skin terms to improve matching
        """
        # Normalize the query formatting first
        normalized_query = self._normalize_query_format(query)
        
        # Check for common weapon types
        weapon_types = ["ak-47", "m4a4", "m4a1-s", "awp", "desert eagle", "deagle", 
                       "glock-18", "knife", "karambit", "bayonet", "butterfly", "gloves"]
        wear_types = ["factory new", "fn", "minimal wear", "mw", "field-tested", "ft", 
                     "well-worn", "ww", "battle-scarred", "bs"]
        
        # Add some common terms if not present
        expanded_terms = []
        
        # Check for specific weapon types
        for weapon in weapon_types:
            if weapon in normalized_query.lower():
                expanded_terms.append(weapon)
        
        # Check for wear conditions
        for wear in wear_types:
            if wear in normalized_query.lower():
                expanded_terms.append(wear)
        
        # Check for gloves
        if "glove" in normalized_query.lower() or "gloves" in normalized_query.lower():
            if not any(term in normalized_query.lower() for term in ["sport gloves", "driver gloves", "specialist gloves", 
                                                         "moto gloves", "hand wraps", "bloodhound gloves"]):
                expanded_terms.append("sport gloves")
        
        # Check for specific patterns
        if "fade" in normalized_query.lower() and not any(w in normalized_query.lower() for w in weapon_types):
            expanded_terms.append("knife")
        
        if "doppler" in normalized_query.lower() and not any(w in normalized_query.lower() for w in weapon_types):
            expanded_terms.append("knife")
            
        # Add expanded terms to query
        if expanded_terms:
            return f"{normalized_query} {' '.join(expanded_terms)}"
        
        return normalized_query
        
    def _normalize_query_format(self, query: str) -> str:
        """
        Normalize query to match CS:GO skin naming conventions
        
        - Convert "weapon skin" to "Weapon | Skin"
        - Handle common typos and abbreviations
        """
        # Dictionary of weapon name mappings (lowercase to proper case)
        weapon_mappings = {
            "ak47": "AK-47",
            "ak": "AK-47",
            "ak-47": "AK-47",
            "m4a4": "M4A4",
            "m4a1s": "M4A1-S",
            "m4a1-s": "M4A1-S",
            "m4": "M4A4",
            "awp": "AWP",
            "desert eagle": "Desert Eagle",
            "deagle": "Desert Eagle",
            "eagle": "Desert Eagle",
            "glock": "Glock-18",
            "glock18": "Glock-18",
            "glock-18": "Glock-18",
            "glocks": "Glock-18",
            "usp": "USP-S",
            "usps": "USP-S",
            "usp-s": "USP-S",
            "p250": "P250",
            "five-seven": "Five-SeveN",
            "fiveseven": "Five-SeveN",
            "cz75": "CZ75-Auto",
            "cz": "CZ75-Auto",
            "tec9": "Tec-9",
            "tec-9": "Tec-9",
            "nova": "Nova",
            "sawed-off": "Sawed-Off",
            "sawedoff": "Sawed-Off",
            "mag7": "MAG-7",
            "mag-7": "MAG-7",
            "xm1014": "XM1014",
            "xm": "XM1014",
            "p90": "P90",
            "mp9": "MP9",
            "mac10": "MAC-10",
            "mac-10": "MAC-10",
            "mp7": "MP7",
            "ump45": "UMP-45",
            "ump-45": "UMP-45",
            "pp-bizon": "PP-Bizon",
            "ppbizon": "PP-Bizon",
            "bizon": "PP-Bizon",
            "negev": "Negev",
            "m249": "M249",
            "galil": "Galil AR",
            "galil ar": "Galil AR",
            "famas": "FAMAS",
            "sg553": "SG 553",
            "sg-553": "SG 553",
            "sg 553": "SG 553",
            "aug": "AUG",
            "ssg08": "SSG 08",
            "ssg-08": "SSG 08",
            "ssg 08": "SSG 08",
            "scout": "SSG 08",
            "g3sg1": "G3SG1",
            "scar20": "SCAR-20",
            "scar-20": "SCAR-20",
        }
        
        # Try to identify weapon and skin parts
        query_lower = query.lower()
        
        # First check if query already has the | format
        if " | " in query:
            return query  # Already in proper format
            
        # Try to match the weapon name at the beginning of the query
        weapon_name = None
        skin_name = None
        
        # Check entire query against weapon mappings first (for queries like "ak-47")
        if query_lower in weapon_mappings:
            return weapon_mappings[query_lower]
            
        # Look for weapons in the query
        for weapon_key, weapon_proper in weapon_mappings.items():
            # Check if query starts with this weapon name
            if query_lower.startswith(weapon_key + " "):
                weapon_name = weapon_proper
                # Extract the rest as the skin name
                skin_name = query[len(weapon_key):].strip()
                break
        
        # If we identified both weapon and skin, format properly
        if weapon_name and skin_name:
            # Capitalize first letter of each word in skin name
            skin_parts = skin_name.split()
            formatted_skin = " ".join(word.capitalize() for word in skin_parts)
            return f"{weapon_name} | {formatted_skin}"
        
        # If we couldn't parse properly, just return original query
        return query

    def load_model(self):
        """Load the sentence transformer model"""
        try:
            cache_dir = Path.home() / ".cache" / "sentence-transformers"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=str(cache_dir),
                device="cpu"
            )
            print(f"Successfully loaded model {self.model_name}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model {self.model_name}: {str(e)}")
            
    def add_skins(self, skins: List[Dict[str, Any]]):
        """Add skins to the search index"""
        if not self.model:
            self.load_model()
            
        # Extract text descriptions for embedding
        texts = []
        for skin in skins:
            desc = f"{skin.get('name', '')} {skin.get('description', '')} {skin.get('rarity', '')}"
            texts.append(desc)
            
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=False)
        
        # Initialize FAISS index if not exists
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            
        # Add to index
        self.index.add(np.array(embeddings).astype('float32'))
        self.skin_data.extend(skins)
        
    def search_with_model(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for skins matching the query"""
        if not self.model or not self.index:
            raise RuntimeError("Search engine not initialized. Call add_skins first.")
            
        # Generate query embedding
        query_embedding = self.model.encode([query], show_progress_bar=False)
        
        # Search in FAISS index
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), 
            k
        )
        
        # Return matching skins
        results = []
        for idx in indices[0]:
            if idx < len(self.skin_data):
                results.append(self.skin_data[idx])
                
        return results


# Initialize a global instance to be imported by other modules
def get_skin_search_engine(data_path: Optional[str] = None) -> SkinSearchEngine:
    """Get or create the skin search engine singleton"""
    if not hasattr(get_skin_search_engine, 'instance') or get_skin_search_engine.instance is None:
        try:
            if data_path is None:
                # Use default path
                current_dir = os.path.dirname(os.path.abspath(__file__))
                data_path = os.path.join(current_dir, "data", "prices_output.json")
                
                # Check if file exists
                if not os.path.exists(data_path):
                    # Try alternative data file
                    alt_path = os.path.join(current_dir, "data", "skinport_data.json")
                    if os.path.exists(alt_path):
                        data_path = alt_path
                    else:
                        print(f"Warning: Could not find default data file at {data_path} or {alt_path}")
            
            # Create and initialize the search engine
            print(f"Initializing search engine with data from: {data_path}")
            engine = SkinSearchEngine(data_path)
            get_skin_search_engine.instance = engine
            
            # Basic validation that the engine is properly initialized
            if not engine.items or not engine.item_names:
                print("Warning: Search engine initialized but no items loaded")
            else:
                print(f"Search engine loaded with {len(engine.item_names)} items")
                
        except Exception as e:
            print(f"Error initializing search engine: {e}")
            # Create a minimal instance without data for graceful fallback
            get_skin_search_engine.instance = SkinSearchEngine()
    
    return get_skin_search_engine.instance 