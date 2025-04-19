import os
import json
import re
from typing import Dict, List, Tuple, Any, Optional
from fuzzywuzzy import process, fuzz

class SimpleSkinSearchEngine:
    """
    A streamlined search engine that prioritizes direct matching and simple fuzzy search
    without requiring heavyweight ML dependencies.
    """
    def __init__(self, data_path: str = None):
        """
        Initialize the search engine with direct matching capabilities
        
        Args:
            data_path: Path to the JSON file with skin data
        """
        self.items = {}
        self.item_names = []
        # Store lowercase versions for case-insensitive matching
        self.item_names_lower = []
        # Special index for weapon types
        self.weapon_type_index = {}
        
        # Initialize data if path is provided
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path: str):
        """Load skin data from JSON file and prepare for search"""
        print(f"Loading skin data from: {data_path}")
        
        # Load the marketplace data JSON
        with open(data_path, 'r', encoding='utf-8') as file:
            marketplace_data = json.load(file)
        
        # Handle different possible JSON structures
        if isinstance(marketplace_data, list):
            # Convert list of items to dictionary with market_hash_name as key
            self.items = {item['market_hash_name']: item for item in marketplace_data}
        else:
            # Handle old format or other structures
            if "marketplace_data" in marketplace_data:
                self.items = marketplace_data.get("marketplace_data", {})
            else:
                self.items = marketplace_data
        
        # Create a list of item names for matching
        self.item_names = list(self.items.keys())
        self.item_names_lower = [name.lower() for name in self.item_names]
        
        # Build weapon type index for faster filtering
        self._build_weapon_index()
        
        print(f"Loaded {len(self.item_names)} CS2 skin items")
    
    def _build_weapon_index(self):
        """Build an index of items by weapon type for efficient filtering"""
        weapon_types = [
            "AK-47", "M4A4", "M4A1-S", "AWP", "Desert Eagle", "USP-S", "Glock-18",
            "P250", "Five-SeveN", "CZ75-Auto", "Tec-9", "Knife", "Karambit", "Bayonet",
            "Butterfly", "Gloves", "P90", "MAC-10", "MP9", "MP7", "UMP-45", "PP-Bizon",
            "Galil AR", "FAMAS", "SG 553", "AUG", "SSG 08", "G3SG1", "SCAR-20"
        ]
        
        # Initialize the index
        self.weapon_type_index = {weapon.lower(): [] for weapon in weapon_types}
        
        # Add item names to the appropriate weapon type lists
        for item_name in self.item_names:
            item_lower = item_name.lower()
            for weapon in weapon_types:
                if weapon.lower() in item_lower:
                    self.weapon_type_index[weapon.lower()].append(item_name)
                    break
    
    def exact_match(self, query: str) -> List[str]:
        """
        Find exact matches for the query string
        
        Args:
            query: The search query
            
        Returns:
            List of matching item names
        """
        query_lower = query.lower()
        
        # First try for an exact match (case-insensitive)
        exact_matches = []
        for i, name_lower in enumerate(self.item_names_lower):
            if query_lower == name_lower:
                exact_matches.append(self.item_names[i])
        
        if exact_matches:
            return exact_matches
            
        # Then try for a contains match
        contains_matches = []
        for i, name_lower in enumerate(self.item_names_lower):
            if query_lower in name_lower:
                contains_matches.append(self.item_names[i])
        
        return contains_matches
    
    def search_by_weapon_and_skin(self, weapon_type: str, skin_name: str) -> List[str]:
        """
        Search for items by specified weapon type and skin name
        
        Args:
            weapon_type: The weapon type to search for
            skin_name: The skin name to search for
            
        Returns:
            List of matching item names
        """
        weapon_lower = weapon_type.lower()
        skin_lower = skin_name.lower()
        
        # Get items for this weapon type
        weapon_items = self.weapon_type_index.get(weapon_lower, [])
        
        # Filter by skin name
        matches = []
        for item_name in weapon_items:
            if skin_lower in item_name.lower():
                matches.append(item_name)
        
        return matches
    
    def search_cheapest_by_weapon(self, weapon_type: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Find the cheapest skins for a specific weapon type
        
        Args:
            weapon_type: The weapon type to search for
            limit: Optional limit on number of results (None = no limit)
            
        Returns:
            List of item data dictionaries with price info, sorted by price
        """
        weapon_lower = weapon_type.lower()
        
        # Get items for this weapon type
        weapon_items = self.weapon_type_index.get(weapon_lower, [])
        if not weapon_items:
            return []
            
        # Extract price data and sort
        price_data = []
        for item_name in weapon_items:
            item_data = self.items[item_name]
            try:
                min_price = float(item_data.get('min_price', '999999'))
                price_data.append({
                    'item_name': item_name,
                    'min_price': min_price,
                    'max_price': float(item_data.get('max_price', '999999')),
                    'suggested_price': float(item_data.get('suggested_price', '999999')),
                    'quantity': item_data.get('quantity', 0),
                    'item_data': item_data
                })
            except (ValueError, TypeError):
                # Skip items with invalid price data
                continue
        
        # Sort by minimum price
        price_data.sort(key=lambda x: x['min_price'])
        
        # Apply limit if specified
        if limit is not None:
            price_data = price_data[:limit]
            
        return price_data
    
    def search_by_price_range(self, weapon_type: str = None, max_price: float = None, min_price: float = 0) -> List[Dict[str, Any]]:
        """
        Find skins within a specific price range, optionally filtered by weapon type
        
        Args:
            weapon_type: Optional weapon type to filter by
            max_price: Maximum price (inclusive)
            min_price: Minimum price (inclusive), defaults to 0
            
        Returns:
            List of item data dictionaries with price info, sorted by price
        """
        # Determine which items to search
        if weapon_type:
            weapon_lower = weapon_type.lower()
            item_names = self.weapon_type_index.get(weapon_lower, [])
        else:
            item_names = self.item_names
            
        # No items to search
        if not item_names:
            return []
            
        # Extract price data
        price_data = []
        for item_name in item_names:
            item_data = self.items[item_name]
            try:
                # Skip stickers, patches, graffiti and cases (they're not weapon skins)
                # Check for these keywords in the item name or category field
                if (("Sticker" in item_name or 
                    "Patch" in item_name or 
                    "Graffiti" in item_name or 
                    "Case" in item_name or 
                    "Container" in item_name or 
                    "Capsule" in item_name or 
                    "Music Kit" in item_name or 
                    "Charm" in item_name) and 
                    not any(weapon in item_name for weapon in [
                        "AK-47", "M4A4", "M4A1-S", "AWP", "Desert Eagle", "USP-S", "Glock-18",
                        "P250", "Five-SeveN", "CZ75-Auto", "Tec-9", "Knife", "Karambit", "Bayonet",
                        "Butterfly", "Gloves", "P90", "MAC-10", "MP9", "MP7", "UMP-45", "PP-Bizon",
                        "Galil AR", "FAMAS", "SG 553", "AUG", "SSG 08", "G3SG1", "SCAR-20"
                    ])):
                    continue
                    
                current_price = float(item_data.get('min_price', '999999'))
                
                # Check if within price range
                if (max_price is None or current_price <= max_price) and current_price >= min_price:
                    price_data.append({
                        'item_name': item_name,
                        'min_price': current_price,
                        'max_price': float(item_data.get('max_price', '999999')),
                        'suggested_price': float(item_data.get('suggested_price', '999999')),
                        'quantity': item_data.get('quantity', 0),
                        'item_data': item_data
                    })
            except (ValueError, TypeError):
                # Skip items with invalid price data
                continue
        
        # Smart sorting based on query type
        if max_price is not None and min_price > 0:
            # For "between X and Y" queries, sort by price ascending
            price_data.sort(key=lambda x: x['min_price'])
        elif max_price is not None:
            # For "under X" queries, sort by price descending (closest to max_price first)
            price_data.sort(key=lambda x: -x['min_price'])
        elif min_price > 0:
            # For "over X" queries, sort by price ascending (closest to min_price first)
            price_data.sort(key=lambda x: x['min_price'])
        else:
            # Default sort by price ascending
            price_data.sort(key=lambda x: x['min_price'])
        
        # Limit results to prevent hitting token limits
        # For price range queries, return at most 15 items
        return price_data[:15]
    
    def fuzzy_search(self, query: str, top_k: int = 10) -> List[Tuple[str, int]]:
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
        
        # Clean and normalize query
        query = query.lower().strip()
        
        # Extract potential weapon type and skin name
        weapon_type = None
        skin_name = None
        
        # Common weapon prefixes
        weapon_prefixes = {
            "ak": "ak-47", "ak47": "ak-47", "ak-47": "ak-47",
            "m4a4": "m4a4", "m4a1s": "m4a1-s", "m4a1-s": "m4a1-s", "m4": "m4a4",
            "awp": "awp", "deagle": "desert eagle", "desert eagle": "desert eagle",
            "glock": "glock-18", "glock18": "glock-18", "glock-18": "glock-18",
            "usp": "usp-s", "usps": "usp-s", "usp-s": "usp-s",
            "p250": "p250", "fiveseven": "five-seven", "five-seven": "five-seven",
            "cz": "cz75-auto", "cz75": "cz75-auto", "cz75-auto": "cz75-auto",
            "tec9": "tec-9", "tec-9": "tec-9"
        }
        
        # Check if query starts with a known weapon prefix
        parts = query.split()
        if parts and parts[0] in weapon_prefixes:
            weapon_type = weapon_prefixes[parts[0]]
            skin_name = " ".join(parts[1:])
            
            # If we identified both weapon and skin, do a more targeted search
            if skin_name:
                weapon_results = self.search_by_weapon_and_skin(weapon_type, skin_name)
                if weapon_results:
                    # Convert to format expected by caller
                    return [(name, 100) for name in weapon_results]
        
        # If direct matching didn't work, fall back to fuzzy matching
        return process.extract(query, self.item_names, limit=top_k)
    
    def detect_price_query(self, query: str) -> Tuple[bool, Optional[float], Optional[float]]:
        """
        Detect if a query is related to price and extract price thresholds
        
        Args:
            query: The search query
            
        Returns:
            Tuple of (is_price_query, max_price, min_price)
        """
        query_lower = query.lower()
        
        # Basic price query detection
        price_keywords = [
            "price", "cost", "value", "worth", "expensive", "cheap", "cheapest", 
            "affordable", "budget", "money", "dollar", "usd", "$"
        ]
        
        is_price_query = any(keyword in query_lower for keyword in price_keywords)
        
        # Detect price range/threshold
        max_price = None
        min_price = None
        
        # Patterns for price thresholds: under/less than $X, over/more than $X, between $X and $Y
        # Match "under $X", "less than $X", "cheaper than $X", "below $X", etc.
        under_patterns = [
            r'under\s*\$?(\d+\.?\d*)',
            r'less than\s*\$?(\d+\.?\d*)',
            r'cheaper than\s*\$?(\d+\.?\d*)',
            r'below\s*\$?(\d+\.?\d*)',
            r'(?:max|maximum)\s*(?:of)?\s*\$?(\d+\.?\d*)',
            r'(?:at most|no more than)\s*\$?(\d+\.?\d*)',
            r'(?:up to|not exceeding)\s*\$?(\d+\.?\d*)'
        ]
        
        # Match "over $X", "more than $X", "above $X", "at least $X", etc.
        over_patterns = [
            r'over\s*\$?(\d+\.?\d*)',
            r'more than\s*\$?(\d+\.?\d*)', 
            r'above\s*\$?(\d+\.?\d*)',
            r'(?:min|minimum)\s*(?:of)?\s*\$?(\d+\.?\d*)',
            r'(?:at least|no less than)\s*\$?(\d+\.?\d*)',
            r'starting (?:at|from)\s*\$?(\d+\.?\d*)'
        ]
        
        # Match "between $X and $Y"
        between_pattern = r'between\s*\$?(\d+\.?\d*)\s*(?:and|to|\-)\s*\$?(\d+\.?\d*)'
        
        # First check for "between" ranges
        between_match = re.search(between_pattern, query_lower)
        if between_match:
            try:
                min_price = float(between_match.group(1))
                max_price = float(between_match.group(2))
                is_price_query = True
            except (ValueError, IndexError):
                pass
        else:
            # Check for upper bound
            for pattern in under_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    try:
                        max_price = float(match.group(1))
                        is_price_query = True
                        break
                    except (ValueError, IndexError):
                        pass
            
            # Check for lower bound
            for pattern in over_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    try:
                        min_price = float(match.group(1))
                        is_price_query = True
                        break
                    except (ValueError, IndexError):
                        pass
        
        # Check if the query contains a simple dollar amount
        if not (max_price or min_price):
            dollar_match = re.search(r'\$?(\d+\.?\d*)', query_lower)
            if dollar_match and ("price" in query_lower or "$" in query_lower):
                try:
                    exact_price = float(dollar_match.group(1))
                    # Use a small range around the exact price
                    max_price = exact_price * 1.1  # 10% above
                    min_price = exact_price * 0.9  # 10% below
                    is_price_query = True
                except (ValueError, IndexError):
                    pass
        
        return is_price_query, max_price, min_price
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for items matching the query and return formatted results
        
        Args:
            query: The search query
            limit: Maximum results to return (None = no limit)
            
        Returns:
            List of result dictionaries with item and price data
        """
        query = query.lower().strip()
        
        # Check for price-related patterns
        is_price_query, max_price, min_price = self.detect_price_query(query)
        
        # Check for specific weapon types
        weapon_names = {
            "ak-47": ["ak47", "ak-47", "ak"],
            "m4a4": ["m4a4"],
            "m4a1-s": ["m4a1s", "m4a1-s", "m4a1"],
            "awp": ["awp"],
            "desert eagle": ["deagle", "desert eagle", "eagle"],
            "glock-18": ["glock", "glock-18", "glock18"],
            "usp-s": ["usp-s", "usps", "usp"],
            "p250": ["p250"],
            "knife": ["knife", "knives"]
        }
        
        detected_weapon = None
        for weapon, aliases in weapon_names.items():
            if any(alias in query for alias in aliases):
                detected_weapon = weapon
                break
                
        # Try to extract skin name if weapon is detected
        skin_name = None
        if detected_weapon:
            # Remove weapon name and price-related terms from query
            clean_query = query
            for alias in weapon_names.get(detected_weapon, []):
                clean_query = clean_query.replace(alias, "")
                
            for term in ["price", "cost", "value", "how much", "cheapest", "expensive"]:
                clean_query = clean_query.replace(term, "")
                
            skin_name = clean_query.strip()
        
        results = []
        
        # Case 1: Price range query with or without weapon type
        if is_price_query and (max_price is not None or min_price is not None):
            # This is a price range query like "under $10" or "more than $50"
            # Results will be limited to 15 items to avoid token limit issues
            price_results = self.search_by_price_range(detected_weapon, max_price, min_price or 0)
            if price_results:
                return price_results
        
        # Case 2: Cheapest weapon skin query (always sort by price)
        if ("cheapest" in query or "lowest price" in query or "least expensive" in query):
            if detected_weapon:
                price_data = self.search_cheapest_by_weapon(detected_weapon, limit=25)  # Increased limit for cheapest queries
                if price_data:
                    return price_data
            else:
                # Generic cheapest query - search all items
                all_items = []
                for item_name in self.item_names:
                    item_data = self.items[item_name]
                    try:
                        min_price = float(item_data.get('min_price', '999999'))
                        all_items.append({
                            'item_name': item_name,
                            'min_price': min_price,
                            'max_price': float(item_data.get('max_price', '999999')),
                            'suggested_price': float(item_data.get('suggested_price', '999999')),
                            'quantity': item_data.get('quantity', 0),
                            'item_data': item_data
                        })
                    except (ValueError, TypeError):
                        continue
                
                # Sort by price and return top 25
                all_items.sort(key=lambda x: x['min_price'])
                return all_items[:25]
                
        # Case 3: Specific weapon + skin query
        if detected_weapon and skin_name:
            matches = self.search_by_weapon_and_skin(detected_weapon, skin_name)
            
            if matches:
                for item_name in matches:
                    item_data = self.items[item_name]
                    try:
                        min_price = float(item_data.get('min_price', '999999'))
                        results.append({
                            'item_name': item_name,
                            'min_price': min_price,
                            'max_price': float(item_data.get('max_price', '999999')),
                            'suggested_price': float(item_data.get('suggested_price', '999999')),
                            'quantity': item_data.get('quantity', 0),
                            'item_data': item_data
                        })
                    except (ValueError, TypeError):
                        continue
                
                # Sort by relevance for skin name match
                if skin_name:
                    results.sort(key=lambda x: -fuzz.partial_ratio(skin_name, x['item_name'].lower()))
                
                return results[:limit] if limit else results
        
        # Case 4: Weapon-only query (no specific skin)
        if detected_weapon and not skin_name:
            # For weapon-only queries, if it's a price query, sort by price
            if is_price_query:
                price_data = self.search_cheapest_by_weapon(detected_weapon, limit=limit)
                if price_data:
                    return price_data
            
            # Otherwise, search all skins for this weapon
            matches = self.weapon_type_index.get(detected_weapon.lower(), [])
            if matches:
                for item_name in matches:
                    item_data = self.items[item_name]
                    try:
                        min_price = float(item_data.get('min_price', '999999'))
                        results.append({
                            'item_name': item_name,
                            'min_price': min_price,
                            'max_price': float(item_data.get('max_price', '999999')),
                            'suggested_price': float(item_data.get('suggested_price', '999999')),
                            'quantity': item_data.get('quantity', 0),
                            'item_data': item_data
                        })
                    except (ValueError, TypeError):
                        continue
                
                # Sort by price if it's a price query, otherwise alphabetically
                if is_price_query:
                    results.sort(key=lambda x: x['min_price'])
                else:
                    results.sort(key=lambda x: x['item_name'])
                
                return results[:limit] if limit else results
        
        # Case 5: Try exact match
        exact_matches = self.exact_match(query)
        if exact_matches:
            for item_name in exact_matches:
                item_data = self.items[item_name]
                try:
                    min_price = float(item_data.get('min_price', '999999'))
                    results.append({
                        'item_name': item_name,
                        'min_price': min_price,
                        'max_price': float(item_data.get('max_price', '999999')),
                        'suggested_price': float(item_data.get('suggested_price', '999999')),
                        'quantity': item_data.get('quantity', 0),
                        'item_data': item_data
                    })
                except (ValueError, TypeError):
                    continue
                    
            # Sort by price if it's a price query
            if is_price_query:
                results.sort(key=lambda x: x['min_price'])
                
            return results[:limit] if limit else results
            
        # Case 6: Fall back to fuzzy search
        fuzzy_results = self.fuzzy_search(query, top_k=limit if limit else 20)
        
        if fuzzy_results:
            for item_name, score in fuzzy_results:
                item_data = self.items[item_name]
                try:
                    min_price = float(item_data.get('min_price', '999999'))
                    results.append({
                        'item_name': item_name,
                        'min_price': min_price,
                        'max_price': float(item_data.get('max_price', '999999')),
                        'suggested_price': float(item_data.get('suggested_price', '999999')),
                        'quantity': item_data.get('quantity', 0),
                        'match_score': score,
                        'item_data': item_data
                    })
                except (ValueError, TypeError):
                    continue
            
            # Sort by fuzzy match score, then by price if it's a price query
            if is_price_query:
                results.sort(key=lambda x: (x['min_price']))
            else:
                results.sort(key=lambda x: (-x.get('match_score', 0)))
                
        return results[:limit] if limit else results
    
    def format_search_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """
        Format search results into a readable string
        
        Args:
            results: List of search result dictionaries
            query: The original search query
            
        Returns:
            Formatted string with search results
        """
        if not results:
            return f"I couldn't find any CS2 skins matching '{query}'. Please try using a more specific name or check your spelling."
        
        # Detect query type
        is_price_query, max_price, min_price = self.detect_price_query(query)
        query_lower = query.lower()
        
        # Determine if this was a specific weapon query
        weapon_names = ["ak-47", "m4a4", "m4a1-s", "awp", "desert eagle", "glock-18", "usp-s", "p250", "knife"]
        detected_weapon = None
        for weapon in weapon_names:
            if weapon in query_lower or weapon.replace("-", "") in query_lower:
                detected_weapon = weapon.upper()
                break
        
        # Check if results might have been limited
        is_limited = (is_price_query and (max_price is not None or min_price is not None) and len(results) == 15)
        
        # Format the header based on query type
        header = f"Found {len(results)} CS2 skin"
        if len(results) != 1:
            header += "s"
        
        # Add weapon info if detected
        if detected_weapon:
            header += f" for {detected_weapon}"
        
        # Add price range info if provided
        if max_price is not None and min_price is not None:
            header += f" between ${min_price:.2f} and ${max_price:.2f}"
        elif max_price is not None:
            header += f" under ${max_price:.2f}"
        elif min_price is not None:
            header += f" over ${min_price:.2f}"
        
        # Add cheapest item summary for price queries
        if is_price_query and len(results) > 0:
            cheapest_item = min(results, key=lambda x: x['min_price'])
            header += f"\nThe cheapest{' ' + detected_weapon if detected_weapon else ''} skin is {cheapest_item['item_name']} at ${cheapest_item['min_price']:.2f}"
        
        # Add note about limited results
        if is_limited:
            header += f"\n\nNote: I've shown the top 15 relevant skins. To see more specific results, try:"
            if not detected_weapon:
                header += f"\n• Adding a specific weapon (like 'AK-47 under ${max_price:.2f}')"
            header += f"\n• Narrowing the price range (like 'between ${max_price-5:.2f} and ${max_price:.2f}')"
            header += f"\n• Specifying a skin name (like '{detected_weapon if detected_weapon else 'AWP'} Asiimov')"
        
        # Format each result
        formatted_results = []
        for item in results:
            item_text = (
                f"{item['item_name']}:\n"
                f"• Skinport Price: ${item['min_price']:.2f}"
            )
            
            if item.get('max_price', 0) != item.get('min_price', 0):
                item_text += f" - ${item['max_price']:.2f}"
                
            item_text += f"\n• Suggested Price: ${item['suggested_price']:.2f}"
            item_text += f"\n• Available: {item['quantity']} items"
            
            formatted_results.append(item_text)
        
        return f"{header}\n\n" + "\n\n".join(formatted_results)


# Singleton instance helper
_instance = None

def get_skin_search_engine(data_path: Optional[str] = None) -> SimpleSkinSearchEngine:
    """Get or create the skin search engine singleton"""
    global _instance
    
    if _instance is None:
        try:
            if data_path is None:
                # Use default path
                current_dir = os.path.dirname(os.path.abspath(__file__))
                data_path = os.path.join(current_dir, "data", "skinport_data.json")
                
                # Check if file exists
                if not os.path.exists(data_path):
                    # Try alternative data file
                    alt_path = os.path.join(current_dir, "data", "prices_output.json")
                    if os.path.exists(alt_path):
                        data_path = alt_path
                    else:
                        print(f"Warning: Could not find default data file at {data_path} or {alt_path}")
            
            # Create and initialize the search engine
            print(f"Initializing simplified search engine with data from: {data_path}")
            engine = SimpleSkinSearchEngine(data_path)
            _instance = engine
            
            # Basic validation that the engine is properly initialized
            if not engine.items or not engine.item_names:
                print("Warning: Search engine initialized but no items loaded")
            else:
                print(f"Search engine loaded with {len(engine.item_names)} items")
                
        except Exception as e:
            print(f"Error initializing search engine: {e}")
            # Create a minimal instance without data for graceful fallback
            _instance = SimpleSkinSearchEngine()
    
    return _instance 