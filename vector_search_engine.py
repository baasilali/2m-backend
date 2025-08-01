import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Any, Optional, Union
import pickle
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorSkinSearchEngine:
    """
    Advanced search engine using vector embeddings for semantic understanding
    of CS2 skin queries with improved context awareness and intent recognition.
    """
    
    def __init__(self, data_path: str = None, embedding_cache_path: str = None, 
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector-based search engine
        
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
        self.intent_classifier = None
        
        # Enhanced search metadata
        self.weapon_categories = {}
        self.skin_patterns = {}
        self.price_ranges = {}
        self.wear_conditions = {}
        
        # Cache paths
        self.embedding_cache_path = embedding_cache_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "data", 
            "vector_embeddings_cache.pkl"
        )
        
        # Load model and initialize
        self._load_model()
        
        if data_path:
            self.load_data(data_path)
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_data(self, data_path: str):
        """Load skin data and create enhanced vector representations"""
        logger.info(f"Loading skin data from: {data_path}")
        
        # Load the marketplace data JSON
        with open(data_path, 'r', encoding='utf-8') as file:
            marketplace_data = json.load(file)
        
        # Handle different JSON structures
        if isinstance(marketplace_data, list):
            self.items = {item['market_hash_name']: item for item in marketplace_data}
        else:
            if "marketplace_data" in marketplace_data:
                self.items = marketplace_data.get("marketplace_data", {})
            else:
                self.items = marketplace_data
        
        self.item_names = list(self.items.keys())
        logger.info(f"Loaded {len(self.item_names)} CS2 skin items")
        
        # Build enhanced metadata indices
        self._build_enhanced_indices()
        
        # Create or load vector embeddings
        self._create_or_load_embeddings()
        
        # Build intent classification system
        self._build_intent_classifier()
    
    def _build_enhanced_indices(self):
        """Build comprehensive indices for different aspects of CS2 skins"""
        
        # Initialize categories
        self.weapon_categories = {}
        self.skin_patterns = {}
        self.price_ranges = {"budget": [], "mid": [], "premium": [], "luxury": []}
        self.wear_conditions = {}
        
        # Define weapon categories with aliases
        weapon_aliases = {
            "ak47": ["ak-47", "ak 47", "kalashnikov"],
            "m4a4": ["m4a4", "m4 a4", "m4"],
            "m4a1s": ["m4a1-s", "m4a1", "m4 silencer"],
            "awp": ["awp", "arctic warfare", "sniper"],
            "glock": ["glock-18", "glock 18", "glock"],
            "usp": ["usp-s", "usp", "silencer pistol"],
            "deagle": ["desert eagle", "deagle", "hand cannon"],
            "knife": ["knife", "blade", "karambit", "bayonet", "butterfly"],
            "gloves": ["gloves", "hand wraps"]
        }
        
        # Build weapon category index
        for category, aliases in weapon_aliases.items():
            self.weapon_categories[category] = []
            for item_name in self.item_names:
                item_lower = item_name.lower()
                if any(alias in item_lower for alias in aliases):
                    self.weapon_categories[category].append(item_name)
        
        # Build price range categories
        for item_name, item_data in self.items.items():
            try:
                price = float(item_data.get('suggested_price', 0))
                if price < 5:
                    self.price_ranges["budget"].append(item_name)
                elif price < 50:
                    self.price_ranges["mid"].append(item_name)
                elif price < 200:
                    self.price_ranges["premium"].append(item_name)
                else:
                    self.price_ranges["luxury"].append(item_name)
            except (ValueError, TypeError):
                self.price_ranges["budget"].append(item_name)
        
        # Build wear condition index
        wear_types = ["Factory New", "Minimal Wear", "Field-Tested", "Well-Worn", "Battle-Scarred"]
        for wear in wear_types:
            self.wear_conditions[wear.lower()] = [
                item for item in self.item_names if wear in item
            ]
    
    def _create_enhanced_search_texts(self) -> List[str]:
        """Create enhanced search texts with semantic context"""
        search_texts = []
        
        for item_name in self.item_names:
            item_data = self.items[item_name]
            
            # Parse item name components
            parts = item_name.split(" | ")
            weapon = parts[0].strip()
            skin_name = parts[1].strip() if len(parts) > 1 else ""
            
            # Extract additional context
            is_stattrak = "stattrak" in item_name.lower()
            wear = self._extract_wear_condition(item_name)
            price = item_data.get('suggested_price', 0)
            
            # Create rich context text for embeddings
            context_elements = [
                weapon,
                skin_name,
                wear if wear else "",
                "StatTrak" if is_stattrak else "regular",
                self._get_price_category(price),
                self._extract_weapon_type(weapon),
                self._extract_skin_quality(skin_name),
                item_name  # Include full name for exact matching
            ]
            
            # Join non-empty elements
            search_text = " ".join([elem for elem in context_elements if elem])
            search_texts.append(search_text)
        
        return search_texts
    
    def _extract_wear_condition(self, item_name: str) -> str:
        """Extract wear condition from item name"""
        wear_patterns = {
            "Factory New": "factory new fn",
            "Minimal Wear": "minimal wear mw", 
            "Field-Tested": "field tested ft",
            "Well-Worn": "well worn ww",
            "Battle-Scarred": "battle scarred bs"
        }
        
        for wear, keywords in wear_patterns.items():
            if wear in item_name:
                return keywords
        return ""
    
    def _get_price_category(self, price: float) -> str:
        """Get semantic price category"""
        try:
            price_val = float(price)
            if price_val < 5:
                return "budget cheap affordable"
            elif price_val < 50:
                return "mid-range moderate"
            elif price_val < 200:
                return "premium expensive"
            else:
                return "luxury high-end exclusive"
        except (ValueError, TypeError):
            return "budget"
    
    def _extract_weapon_type(self, weapon: str) -> str:
        """Extract weapon type category"""
        weapon_lower = weapon.lower()
        
        if any(pistol in weapon_lower for pistol in ["glock", "usp", "p250", "cz75", "tec-9", "five-seven", "desert eagle"]):
            return "pistol sidearm"
        elif any(rifle in weapon_lower for rifle in ["ak-47", "m4a4", "m4a1-s", "galil", "famas", "aug", "sg 553"]):
            return "rifle assault"
        elif any(sniper in weapon_lower for sniper in ["awp", "ssg 08", "g3sg1", "scar-20"]):
            return "sniper rifle"
        elif "knife" in weapon_lower or any(knife in weapon_lower for knife in ["karambit", "bayonet", "butterfly"]):
            return "knife melee blade"
        elif "gloves" in weapon_lower:
            return "gloves hand wraps"
        elif any(smg in weapon_lower for smg in ["mac-10", "mp9", "mp7", "ump-45", "p90", "pp-bizon"]):
            return "smg submachine"
        else:
            return "weapon"
    
    def _extract_skin_quality(self, skin_name: str) -> str:
        """Extract skin quality indicators"""
        skin_lower = skin_name.lower()
        
        quality_indicators = {
            "fade": "gradient fade colorful",
            "doppler": "doppler phase pattern",
            "tiger tooth": "tiger striped",
            "marble fade": "marble colorful fade",
            "case hardened": "blue gem pattern",
            "crimson web": "web spider red",
            "slaughter": "pattern diamond angel",
            "safari mesh": "camo military",
            "urban masked": "urban tactical"
        }
        
        for pattern, keywords in quality_indicators.items():
            if pattern in skin_lower:
                return keywords
                
        return "skin pattern design"
    
    def _create_or_load_embeddings(self):
        """Create vector embeddings or load from cache"""
        if os.path.exists(self.embedding_cache_path):
            try:
                with open(self.embedding_cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                if (sorted(cache_data['item_names']) == sorted(self.item_names) and 
                    cache_data['model_name'] == self.model_name):
                    self.embeddings = cache_data['embeddings']
                    self._create_faiss_index()
                    logger.info(f"Loaded embeddings from cache for {len(self.item_names)} items")
                    return
            except Exception as e:
                logger.warning(f"Error loading embedding cache: {e}")
        
        # Create new embeddings
        logger.info("Creating vector embeddings...")
        search_texts = self._create_enhanced_search_texts()
        
        # Generate embeddings in batches for efficiency
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(search_texts), batch_size):
            batch = search_texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=True)
            all_embeddings.append(batch_embeddings)
        
        self.embeddings = np.vstack(all_embeddings)
        
        # Create FAISS index
        self._create_faiss_index()
        
        # Save to cache
        self._save_embeddings_cache()
        
        logger.info(f"Created embeddings for {len(self.item_names)} items")
    
    def _create_faiss_index(self):
        """Create FAISS index for fast similarity search"""
        if self.embeddings is not None:
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
            
            logger.info(f"Created FAISS index with {self.index.ntotal} vectors")
    
    def _save_embeddings_cache(self):
        """Save embeddings to cache file"""
        try:
            os.makedirs(os.path.dirname(self.embedding_cache_path), exist_ok=True)
            cache_data = {
                'embeddings': self.embeddings,
                'item_names': self.item_names,
                'model_name': self.model_name
            }
            with open(self.embedding_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info("Saved embeddings to cache")
        except Exception as e:
            logger.warning(f"Error saving embeddings cache: {e}")
    
    def _build_intent_classifier(self):
        """Build intent classification for different types of queries"""
        self.intent_patterns = {
            'price_range': [
                r'under \$?(\d+)',
                r'below \$?(\d+)', 
                r'cheaper than \$?(\d+)',
                r'less than \$?(\d+)',
                r'between \$?(\d+) and \$?(\d+)',
                r'from \$?(\d+) to \$?(\d+)',
                r'budget',
                r'cheap',
                r'affordable'
            ],
            'weapon_specific': [
                r'(ak-47|ak47|kalashnikov)',
                r'(m4a4|m4a1-s|m4)',
                r'(awp|sniper)',
                r'(glock|usp|deagle|pistol)',
                r'(knife|blade|karambit|bayonet)'
            ],
            'quality_search': [
                r'(factory new|fn)',
                r'(minimal wear|mw)',
                r'(field-tested|ft)',
                r'(stattrak|stat)',
                r'(best|top|highest quality)'
            ],
            'comparison': [
                r'compare',
                r'vs|versus',
                r'difference between',
                r'which is better'
            ]
        }
    
    def classify_intent(self, query: str) -> Dict[str, Any]:
        """Classify the intent of a search query"""
        query_lower = query.lower()
        intents = {}
        
        # Check for price-related intent
        for pattern in self.intent_patterns['price_range']:
            match = re.search(pattern, query_lower)
            if match:
                intents['price_range'] = True
                if len(match.groups()) == 1:
                    intents['max_price'] = float(match.group(1))
                elif len(match.groups()) == 2:
                    intents['min_price'] = float(match.group(1))
                    intents['max_price'] = float(match.group(2))
                break
        
        # Check for weapon-specific intent
        for pattern in self.intent_patterns['weapon_specific']:
            if re.search(pattern, query_lower):
                intents['weapon_specific'] = True
                break
        
        # Check for quality-specific intent
        for pattern in self.intent_patterns['quality_search']:
            if re.search(pattern, query_lower):
                intents['quality_specific'] = True
                break
        
        # Check for comparison intent
        for pattern in self.intent_patterns['comparison']:
            if re.search(pattern, query_lower):
                intents['comparison'] = True
                break
        
        return intents
    
    def vector_search(self, query: str, limit: int = 10, similarity_threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Perform vector-based semantic search"""
        if self.model is None or self.index is None:
            raise ValueError("Search engine not properly initialized")
        
        # Classify query intent
        intent = self.classify_intent(query)
        
        # Enhance query with context
        enhanced_query = self._enhance_query_with_context(query, intent)
        
        # Generate query embedding
        query_embedding = self.model.encode([enhanced_query])
        faiss.normalize_L2(query_embedding)
        
        # Search with FAISS
        scores, indices = self.index.search(query_embedding, min(limit * 3, len(self.item_names)))
        
        # Filter and rank results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if score >= similarity_threshold:
                item_name = self.item_names[idx]
                
                # Apply intent-based filtering
                if self._passes_intent_filter(item_name, intent):
                    results.append((item_name, float(score)))
        
        # Sort by score and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def _enhance_query_with_context(self, query: str, intent: Dict[str, Any]) -> str:
        """Enhance query with contextual information"""
        enhanced_parts = [query]
        
        # Add CS2 context if not present
        if not any(keyword in query.lower() for keyword in ['cs2', 'counter-strike', 'skin']):
            enhanced_parts.append("CS2 skin")
        
        # Add intent-specific context
        if intent.get('price_range'):
            enhanced_parts.append("price range budget")
        
        if intent.get('weapon_specific'):
            enhanced_parts.append("weapon specific")
        
        if intent.get('quality_specific'):
            enhanced_parts.append("condition wear quality")
        
        return " ".join(enhanced_parts)
    
    def _passes_intent_filter(self, item_name: str, intent: Dict[str, Any]) -> bool:
        """Check if item passes intent-based filters"""
        item_data = self.items[item_name]
        
        # Price range filter
        if intent.get('price_range'):
            try:
                item_price = float(item_data.get('suggested_price', 0))
                if intent.get('max_price') and item_price > intent['max_price']:
                    return False
                if intent.get('min_price') and item_price < intent['min_price']:
                    return False
            except (ValueError, TypeError):
                pass
        
        return True
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Main search method that returns formatted results"""
        try:
            # Perform vector search
            vector_results = self.vector_search(query, limit)
            
            # Format results with item data
            formatted_results = []
            for item_name, score in vector_results:
                item_data = self.items[item_name].copy()
                item_data['relevance_score'] = score
                item_data['item_name'] = item_name
                formatted_results.append(item_data)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    def format_search_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """Format search results for display"""
        if not results:
            return f"No CS2 skins found matching '{query}'. Try a more general search term."
        
        # Classify query to determine formatting
        intent = self.classify_intent(query)
        
        output = []
        output.append(f"Found {len(results)} CS2 skins matching '{query}':\n")
        
        # Group by relevance if many results
        if len(results) > 5:
            high_relevance = [r for r in results if r.get('relevance_score', 0) > 0.8]
            if high_relevance:
                output.append("🎯 **Highly Relevant Results:**")
                for item in high_relevance[:3]:
                    output.append(self._format_single_item(item))
                output.append("")
        
        # Show top results
        for i, item in enumerate(results[:limit], 1):
            output.append(f"{i}. {self._format_single_item(item)}")
        
        # Add search tips if many results
        if len(results) >= limit:
            output.append(f"\n💡 Showing top {limit} results. Try more specific terms to narrow your search.")
        
        return "\n".join(output)
    
    def _format_single_item(self, item: Dict[str, Any]) -> str:
        """Format a single item for display"""
        name = item.get('item_name', 'Unknown')
        price = item.get('suggested_price', 'N/A')
        
        try:
            price_val = float(price)
            price_str = f"${price_val:.2f}"
        except (ValueError, TypeError):
            price_str = "Price N/A"
        
        return f"**{name}** - {price_str}"

# Factory function to get the search engine instance
def get_vector_skin_search_engine():
    """Get a configured instance of the vector search engine"""
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "skinport_data.json")
    return VectorSkinSearchEngine(data_path=data_path) 