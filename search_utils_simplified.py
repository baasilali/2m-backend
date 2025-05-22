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
        
        # Also create StatTrak-specific indexes for faster filtering
        self._build_stattrak_index()
        
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
            
            # First check for StatTrak variants - they have the weapon name after the StatTrak prefix
            if "stattrak™" in item_lower or "stattrak" in item_lower:
                for weapon in weapon_types:
                    weapon_lower = weapon.lower()
                    if weapon_lower in item_lower:
                        self.weapon_type_index[weapon_lower].append(item_name)
                        break
            else:
                # Then check regular weapons
                for weapon in weapon_types:
                    weapon_lower = weapon.lower()
                    if weapon_lower in item_lower:
                        self.weapon_type_index[weapon_lower].append(item_name)
                        break
    
    def _build_stattrak_index(self):
        """Build an index for StatTrak items to allow faster filtering"""
        # Create separate lists for StatTrak and non-StatTrak items
        self.stattrak_items = []
        self.non_stattrak_items = []
        
        # Also create a mapping from non-StatTrak to StatTrak versions of the same skin
        self.stattrak_mapping = {}
        
        for item_name in self.item_names:
            item_lower = item_name.lower()
            
            if "stattrak™" in item_lower or "stattrak" in item_lower:
                self.stattrak_items.append(item_name)
                
                # Try to find the non-StatTrak version by removing "StatTrak™ " or "StatTrak "
                non_st_name = item_name.replace("StatTrak™ ", "").replace("StatTrak ", "")
                self.stattrak_mapping[non_st_name] = item_name
            else:
                self.non_stattrak_items.append(item_name)
    
    def exact_match(self, query: str) -> List[str]:
        """
        Find exact matches for the query string with improved accuracy
        
        Args:
            query: The search query
            
        Returns:
            List of matching item names
        """
        query_lower = query.lower().strip()
        
        # Check for StatTrak variants
        is_stattrak = ("stattrak™" in query_lower or "stattrak" in query_lower or 
                      "stat trak" in query_lower or "stat-trak" in query_lower)
        
        # 1. First try for a perfect match (case-insensitive)
        exact_matches = []
        for i, name_lower in enumerate(self.item_names_lower):
            if query_lower == name_lower:
                exact_matches.append(self.item_names[i])
            # If query includes StatTrak but doesn't mention trademark symbol, still match with ™
            elif is_stattrak and "stattrak" in query_lower and "stattrak™" in name_lower and query_lower.replace("stattrak", "stattrak™") == name_lower:
                exact_matches.append(self.item_names[i])
        
        if exact_matches:
            return exact_matches
            
        # 2. Then try parsing the query into components (weapon, skin, wear)
        parsed_matches = self._match_by_parsed_components(query_lower)
        if parsed_matches:
            return parsed_matches
            
        # 3. Then try for a prefix match (query is the start of an item name)
        prefix_matches = []
        for i, name_lower in enumerate(self.item_names_lower):
            if name_lower.startswith(query_lower):
                prefix_matches.append(self.item_names[i])
            # If query includes StatTrak but doesn't mention trademark symbol, handle that
            elif is_stattrak and "stattrak" in query_lower and "stattrak™" in name_lower and name_lower.startswith(query_lower.replace("stattrak", "stattrak™")):
                prefix_matches.append(self.item_names[i])
        
        if prefix_matches:
            return prefix_matches
            
        # 4. Finally, try for a contains match
        contains_matches = []
        for i, name_lower in enumerate(self.item_names_lower):
            if query_lower in name_lower:
                contains_matches.append(self.item_names[i])
            # If query includes StatTrak but doesn't mention trademark symbol, still match with ™
            elif is_stattrak and "stattrak" in query_lower and "stattrak™" in name_lower and query_lower.replace("stattrak", "stattrak™") in name_lower:
                contains_matches.append(self.item_names[i])
        
        return contains_matches
    
    def _match_by_parsed_components(self, query_lower: str) -> List[str]:
        """
        Parse the query into components (weapon, skin, wear) and match accordingly
        
        Args:
            query_lower: The lowercase search query
            
        Returns:
            List of matching item names
        """
        # Try to identify weapon, skin, and wear condition
        weapon_type = None
        skin_name = None
        wear_condition = None
        is_stattrak = False
        
        # Check for StatTrak
        if any(term in query_lower for term in ["stattrak™", "stattrak", "stat trak", "stat-trak", "stattrack", "st"]):
            is_stattrak = True
            # Remove StatTrak terms for cleaner parsing
            for term in ["stattrak™", "stattrak", "stat trak", "stat-trak", "stattrack", "st"]:
                query_lower = query_lower.replace(term, "").strip()
        
        # Common CS2 weapon names
        weapon_names = {
            "ak-47": ["ak47", "ak-47", "ak"],
            "m4a4": ["m4a4"],
            "m4a1-s": ["m4a1s", "m4a1-s", "m4a1"],
            "awp": ["awp"],
            "desert eagle": ["deagle", "desert eagle", "eagle"],
            "glock-18": ["glock", "glock-18", "glock18"],
            "usp-s": ["usp-s", "usps", "usp"],
            "p250": ["p250"],
            "knife": ["knife"],
            "karambit": ["karambit"]
            # Add more weapons as needed
        }
        
        # Try to extract weapon type
        for weapon, aliases in weapon_names.items():
            for alias in aliases:
                if alias in query_lower:
                    weapon_type = weapon
                    # Remove weapon part from query
                    query_lower = query_lower.replace(alias, "").strip()
                    break
            if weapon_type:
                break
        
        # Check for wear conditions
        wear_conditions = {
            "factory new": ["factory new", "fn"],
            "minimal wear": ["minimal wear", "mw"],
            "field-tested": ["field-tested", "field tested", "ft"],
            "well-worn": ["well-worn", "well worn", "ww"],
            "battle-scarred": ["battle-scarred", "battle scarred", "bs"]
        }
        
        for wear, aliases in wear_conditions.items():
            for alias in aliases:
                if alias in query_lower:
                    wear_condition = wear
                    # Remove wear part from query
                    query_lower = query_lower.replace(alias, "").strip()
                    break
            if wear_condition:
                break
        
        # What remains should be the skin name (or part of it)
        skin_name = query_lower.strip()
        
        # Skip if we couldn't extract any components
        if not (weapon_type or skin_name or wear_condition):
            return []
        
        # Find matching items based on extracted components
        matches = []
        for item_name in self.item_names:
            item_lower = item_name.lower()
            
            # Check for StatTrak match
            if is_stattrak and not ("stattrak™" in item_lower or "stattrak" in item_lower):
                continue
                
            # Check for weapon type match
            if weapon_type and weapon_type not in item_lower:
                continue
                
            # Check for skin name match if we extracted one
            if skin_name and skin_name not in item_lower:
                continue
                
            # Check for wear condition match
            if wear_condition and wear_condition not in item_lower:
                continue
                
            # If we got here, the item matches all the criteria we extracted
            matches.append(item_name)
        
        return matches
    
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
        skin_lower = skin_name.lower().strip()
        
        # Get items for this weapon type
        weapon_items = self.weapon_type_index.get(weapon_lower, [])
        
        # Check for StatTrak keyword with more variations
        is_stattrak = any(term in skin_lower for term in ["stattrak™", "stattrak", "stat trak", "stat-trak", "stattrack", "st"])
        
        # Check for wear conditions
        wear_conditions = {
            "factory new": ["factory new", "fn"],
            "minimal wear": ["minimal wear", "mw"],
            "field-tested": ["field-tested", "field tested", "ft"],
            "well-worn": ["well-worn", "well worn", "ww"],
            "battle-scarred": ["battle-scarred", "battle scarred", "bs"]
        }
        
        detected_wear = None
        for wear, aliases in wear_conditions.items():
            if any(alias in skin_lower for alias in aliases):
                detected_wear = wear
                # Remove wear condition from skin name for better matching
                for alias in aliases:
                    skin_lower = skin_lower.replace(alias, "").strip()
                break
        
        # Remove StatTrak terms from the skin name for better matching
        clean_skin_lower = skin_lower
        for term in ["stattrak™", "stattrak", "stat trak", "stat-trak", "stattrack", "st"]:
            clean_skin_lower = clean_skin_lower.replace(term, "").strip()
        
        # Filter by skin name, making sure not to filter out StatTrak variants
        matches = []
        for item_name in weapon_items:
            item_lower = item_name.lower()
            
            # Check if the item is a StatTrak variant
            item_is_stattrak = "stattrak™" in item_lower or "stattrak" in item_lower
            
            # Skip non-StatTrak items if StatTrak was explicitly requested
            if is_stattrak and not item_is_stattrak:
                continue
                
            # Clean the item name for matching (remove StatTrak designation)
            clean_item_name = item_lower.replace("stattrak™ ", "").replace("stattrak ", "")
            
            # Check if the skin name is in the cleaned item name
            if clean_skin_lower in clean_item_name:
                # If wear condition was specified, check if it matches
                if detected_wear is None or detected_wear.lower() in item_lower:
                    matches.append(item_name)
        
        # If we didn't find any matches with an exact substring match, try using fuzzy matching
        if not matches and clean_skin_lower:
            # Get just the skin names by removing weapon name and wear condition
            skin_names = []
            for item_name in weapon_items:
                item_lower = item_name.lower()
                # Skip non-StatTrak items if StatTrak was explicitly requested
                if is_stattrak and not ("stattrak™" in item_lower or "stattrak" in item_lower):
                    continue
                    
                # Extract the skin name part (between | and ()
                parts = item_lower.split("|")
                if len(parts) >= 2:
                    # Get the skin name - text after | and before wear condition
                    skin_part = parts[1].split("(")[0].strip()
                    
                    # If wear condition was specified, check if it matches
                    if detected_wear is None or detected_wear.lower() in item_lower:
                        skin_names.append((item_name, skin_part))
            
            # Use fuzzy matching to find the closest match
            if skin_names:
                # Extract just the skin names for matching
                skin_parts = [name[1] for name in skin_names]
                # Find the best matches
                matches_with_scores = process.extract(clean_skin_lower, skin_parts, limit=10)
                
                # Only keep good quality matches (score >= 75)
                good_matches = [(skin_names[skin_parts.index(name)][0], score) for name, score in matches_with_scores if score >= 75]
                
                # Sort by score (descending) and extract just the item names
                good_matches.sort(key=lambda x: -x[1])
                matches = [name for name, _ in good_matches]
        
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
        
        # Sort by minimum price (ascending for cheapest)
        price_data.sort(key=lambda x: x['min_price'])
        
        # Apply limit if specified
        if limit is not None:
            price_data = price_data[:limit]
            
        return price_data
    
    def search_most_expensive_by_weapon(self, weapon_type: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Find the most expensive skins for a specific weapon type
        
        Args:
            weapon_type: The weapon type to search for
            limit: Optional limit on number of results (None = no limit)
            
        Returns:
            List of item data dictionaries with price info, sorted by price (highest first)
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
        
        # Sort by minimum price (descending for most expensive)
        price_data.sort(key=lambda x: x['min_price'], reverse=True)
        
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
                        "Galil AR", "FAMAS", "SG 553", "AUG", "SSG 08", "G3SG1", "SCAR-20", "Dual Berettas",
                        "R8 Revolver", "P2000", "MP5-SD", "MAG-7", "Nova", "Sawed-Off", "XM1014", "M249", "Negev",
                        "Bowie", "Classic Knife", "Falchion", "Flip", "Gut", "Huntsman", "Kukri", "M9 Bayonet",
                        "Navaja", "Nomad", "Paracord", "Shadow Daggers", "Skeleton", "Stiletto", "Survival", "Talon",
                        "Ursus", "Zeus x27"

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
        
        # Check for StatTrak keyword with more variations
        is_stattrak = any(term in query for term in ["stattrak™", "stattrak", "stat trak", "stat-trak", "stattrack", "st"])
        
        # Normalize StatTrak notation in the query for better matching
        normalized_query = query
        for st_term in ["stat trak", "stat-trak", "stattrack", "st"]:
            if st_term in normalized_query:
                normalized_query = normalized_query.replace(st_term, "stattrak")
        
        # Correct common misspellings of skin names
        skin_corrections = {
            "autorinic": "autotronic",
            "autronic": "autotronic",
            "autoronic": "autotronic",
            "ultrvoilet": "ultraviolet",
            "ultraviolt": "ultraviolet",
            "doplar": "doppler",
            "doplr": "doppler",
            "marbl": "marble",
            "marbel": "marble",
            "marblefade": "marble fade",
            "tigertoot": "tiger tooth",
            "tiger toot": "tiger tooth",
            "casehardened": "case hardened",
            "case-hardened": "case hardened",
            "crim web": "crimson web",
            "crimsonweb": "crimson web",
            "blu steel": "blue steel",
            "damascus": "damascus steel",
            "rust": "rust coat", 
            "gamma dopler": "gamma doppler",
            "gamma-doppler": "gamma doppler"
        }
        
        # Apply spelling corrections to the query
        for misspelling, correction in skin_corrections.items():
            if misspelling in normalized_query:
                normalized_query = normalized_query.replace(misspelling, correction)
                print(f"Corrected '{misspelling}' to '{correction}' in search query")
                
        # Check for wear conditions
        wear_conditions = {
            "factory new": ["factory new", "fn"],
            "minimal wear": ["minimal wear", "mw"],
            "field-tested": ["field-tested", "field tested", "ft"],
            "well-worn": ["well-worn", "well worn", "ww"],
            "battle-scarred": ["battle-scarred", "battle scarred", "bs"]
        }
        
        detected_wear = None
        for wear, aliases in wear_conditions.items():
            if any(alias in normalized_query for alias in aliases):
                detected_wear = wear
                break
        
        # Extract potential weapon type and skin name
        weapon_type = None
        skin_name = None
        
        # Extract exact skin name if present (text after weapon type)
        skin_patterns = [
            "autotronic", "lore", "doppler", "gamma doppler", "marble fade", "tiger tooth", 
            "fade", "crimson web", "slaughter", "case hardened", "ultraviolet", "night", 
            "blue steel", "damascus steel", "rust coat", "scorched", "forest ddpat", 
            "urban masked", "stained", "safari mesh", "boreal forest"
        ]
        
        detected_skin = None
        for pattern in skin_patterns:
            if pattern in normalized_query:
                detected_skin = pattern
                break
        
        # Common weapon prefixes
        weapon_prefixes = {
            "ak": "ak-47", "ak47": "ak-47", "ak-47": "ak-47",
            "m4a4": "m4a4", "m4a1s": "m4a1-s", "m4a1-s": "m4a1-s", "m4": "m4a4",
            "awp": "awp", "deagle": "desert eagle", "desert eagle": "desert eagle",
            "glock": "glock-18", "glock18": "glock-18", "glock-18": "glock-18",
            "usp": "usp-s", "usps": "usp-s", "usp-s": "usp-s",
            "p250": "p250", "fiveseven": "five-seven", "five-seven": "five-seven",
            "cz": "cz75-auto", "cz75": "cz75-auto", "cz75-auto": "cz75-auto",
            "tec9": "tec-9", "tec-9": "tec-9",
            "karambit": "karambit", "bayonet": "bayonet", "butterfly": "butterfly knife",
            "m9": "m9 bayonet", "m9 bayonet": "m9 bayonet", "flip": "flip knife", 
            "gut": "gut knife", "falchion": "falchion knife", "shadow": "shadow daggers",
            "huntsman": "huntsman knife", "bowie": "bowie knife", "daggers": "shadow daggers",
            "knife": "knife"
        }
        
        # Check if query starts with a known weapon prefix
        parts = normalized_query.split()
        if parts and parts[0] in weapon_prefixes:
            weapon_type = weapon_prefixes[parts[0]]
            skin_name = " ".join(parts[1:])
            
            # If we identified both weapon and skin, do a more targeted search
            if skin_name:
                # Look for exact matches first
                if detected_skin and detected_wear:
                    # Try a very targeted search with weapon + skin pattern + wear condition
                    targeted_query = f"{weapon_type} {detected_skin} {detected_wear}"
                    # Prepend StatTrak if needed
                    if is_stattrak:
                        targeted_query = f"stattrak {targeted_query}"
                    
                    # Try to find exact matches for this specific combination
                    exact_matches = []
                    for item_name in self.item_names:
                        item_lower = item_name.lower()
                        if (weapon_type in item_lower and 
                            detected_skin in item_lower and 
                            detected_wear in item_lower and
                            (not is_stattrak or "stattrak" in item_lower)):
                            exact_matches.append(item_name)
                    
                    if exact_matches:
                        return [(name, 100) for name in exact_matches]
                        
                # If specific exact match didn't work, try general weapon+skin search    
                weapon_results = self.search_by_weapon_and_skin(weapon_type, skin_name)
                if weapon_results:
                    # If stattrak is specified, prioritize stattrak items
                    if is_stattrak:
                        stattrak_results = [item for item in weapon_results if "stattrak™" in item.lower() or "stattrak" in item.lower()]
                        if stattrak_results:
                            return [(name, 100) for name in stattrak_results]
                    
                    # Convert to format expected by caller
                    return [(name, 100) for name in weapon_results]
        
        # Special case for Karambit: check if it appears anywhere in the query
        if "karambit" in normalized_query and not weapon_type:
            weapon_type = "karambit"
            # Try to extract the skin name
            skin_name = normalized_query.replace("karambit", "").strip()
            
            if is_stattrak:
                for term in ["stattrak™", "stattrak", "stat trak", "stat-trak", "stattrack", "st"]:
                    skin_name = skin_name.replace(term, "").strip()
            
            if detected_skin and detected_wear:
                # Try a very targeted search with weapon + skin pattern + wear condition
                targeted_query = f"karambit {detected_skin} {detected_wear}"
                # Prepend StatTrak if needed
                if is_stattrak:
                    targeted_query = f"stattrak {targeted_query}"
                
                # Try to find exact matches for this specific combination
                exact_matches = []
                for item_name in self.item_names:
                    item_lower = item_name.lower()
                    if (("karambit" in item_lower) and 
                        detected_skin in item_lower and 
                        detected_wear in item_lower and
                        (not is_stattrak or "stattrak" in item_lower)):
                        exact_matches.append(item_name)
                
                if exact_matches:
                    return [(name, 100) for name in exact_matches]
            
            # If exact pattern+wear didn't work, try just the skin name
            if skin_name:
                weapon_results = self.search_by_weapon_and_skin("karambit", skin_name)
                if weapon_results:
                    # If stattrak is specified, prioritize stattrak items
                    if is_stattrak:
                        stattrak_results = [item for item in weapon_results if "stattrak™" in item.lower() or "stattrak" in item.lower()]
                        if stattrak_results:
                            return [(name, 100) for name in stattrak_results]
                    
                    # Convert to format expected by caller
                    return [(name, 100) for name in weapon_results]
        
        # If direct matching didn't work, fall back to fuzzy matching
        # but prioritize StatTrak items if specified
        if is_stattrak:
            # First try to match against only StatTrak items
            stattrak_results = process.extract(normalized_query, self.stattrak_items, limit=top_k)
            
            # If we have good matches, return them
            if stattrak_results and stattrak_results[0][1] > 80:
                return stattrak_results
                
        # Fall back to regular fuzzy matching
        return process.extract(normalized_query, self.item_names, limit=top_k)
    
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
            "affordable", "budget", "money", "dollar", "usd", "$", "most expensive",
            "highest price", "priciest"
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
    
    def hierarchical_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform search using a hierarchical approach that prioritizes exact matches over fuzzy matches
        
        Args:
            query: The search query
            limit: Maximum results to return
            
        Returns:
            List of result dictionaries with item and price data
        """
        query = query.lower().strip()
        
        # Step 1: Try exact matching first
        exact_matches = self.exact_match(query)
        
        # Step 2: If no exact matches, try parsing the query and matching by components
        if not exact_matches:
            exact_matches = self._match_by_parsed_components(query)
        
        # Step 3: If still no matches, try fuzzy matching
        if not exact_matches and query:
            fuzzy_results = self.fuzzy_search(query, top_k=limit)
            # Only use fuzzy results with good scores (> 75)
            good_fuzzy_matches = [name for name, score in fuzzy_results if score > 75]
            exact_matches = good_fuzzy_matches
        
        # Convert exact match item names to full result dictionaries with price info
        results = []
        for item_name in exact_matches:
            if item_name in self.items:
                item_data = self.items[item_name]
                try:
                    min_price = float(item_data.get('min_price', '999999'))
                    results.append({
                        'item_name': item_name,
                        'min_price': min_price,
                        'max_price': float(item_data.get('max_price', '999999')),
                        'suggested_price': float(item_data.get('suggested_price', '999999')),
                        'quantity': item_data.get('quantity', 0),
                        'item_data': item_data,
                        'match_score': 100  # Exact matches get top score
                    })
                except (ValueError, TypeError):
                    # Skip items with invalid price data
                    continue
        
        # If we still don't have results, implement fallback logic here
        # instead of calling self.search() which would create circular reference
        if not results:
            # Apply spelling corrections to the query
            corrected_query = self._correct_spelling(query)
            if corrected_query != query:
                print(f"Corrected query: '{query}' → '{corrected_query}'")
                query = corrected_query
            
            # Check for specific price-based queries and use specialized handling
            is_price_query, max_price, min_price = self.detect_price_query(query)
            if is_price_query and (max_price is not None or min_price is not None):
                # Try the price range search method
                # Extract weapon type if present
                weapon_type = None
                for key in self.weapon_type_index.keys():
                    if key in query:
                        weapon_type = key
                        break
                    
                price_results = self.search_by_price_range(weapon_type, max_price, min_price or 0)
                if price_results:
                    return price_results[:limit]
        
        return results[:limit]

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for items matching the query and return formatted results
        
        Args:
            query: The search query
            limit: Maximum results to return (None = no limit)
            
        Returns:
            List of result dictionaries with item and price data
        """
        # To avoid circular reference, use a flag to control how we process
        from_hierarchical = getattr(self, '_from_hierarchical', False)
        self._from_hierarchical = False
        
        # If not called from hierarchical_search, use hierarchical_search
        if not from_hierarchical:
            self._from_hierarchical = True
            hierarchical_results = self.hierarchical_search(query, limit=limit)
            if hierarchical_results:
                return hierarchical_results
        
        # If hierarchical search didn't work or we're already in hierarchical search,
        # continue with the original search logic
        query = query.lower().strip()
        
        # Apply spelling corrections to the query
        corrected_query = self._correct_spelling(query)
        if corrected_query != query:
            print(f"Corrected query: '{query}' → '{corrected_query}'")
            query = corrected_query
        
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
            "knife": ["knife", "knives"],
            "karambit": ["karambit"]
        }
        
        detected_weapon = None
        for weapon, aliases in weapon_names.items():
            if any(alias in query for alias in aliases):
                detected_weapon = weapon
                break
        
        # Check for StatTrak keyword with more variations
        is_stattrak = any(term in query for term in ["stattrak™", "stattrak", "stat trak", "stat-trak", "stattrack", "st"])
        
        # Try to extract skin name if weapon is detected
        skin_name = None
        if detected_weapon:
            # Remove weapon name and price-related terms from query
            clean_query = query
            for alias in weapon_names.get(detected_weapon, []):
                clean_query = clean_query.replace(alias, "")
                
            for term in ["price", "cost", "value", "how much", "cheapest", "expensive", 
                         "stattrak™", "stattrak", "stat trak", "stat-trak", "stattrack", "st"]:
                clean_query = clean_query.replace(term, "")
                
            skin_name = clean_query.strip()
        
        results = []
        
        # Case 1: Price range query with or without weapon type
        if is_price_query and (max_price is not None or min_price is not None):
            # This is a price range query like "under $10" or "more than $50"
            # Results will be limited to 15 items to avoid token limit issues
            price_results = self.search_by_price_range(detected_weapon, max_price, min_price or 0)
            if price_results:
                # Filter for StatTrak if specified
                if is_stattrak:
                    price_results = [r for r in price_results if "stattrak™" in r['item_name'].lower() or "stattrak" in r['item_name'].lower()]
                if price_results:  # Make sure we still have results after filtering
                    return price_results
                else:
                    # If filtering removed all results, fall through to other search methods
                    pass
        
        # Case 2: Cheapest weapon skin query (always sort by price)
        if "cheapest" in query or "lowest price" in query or "least expensive" in query:
            if detected_weapon:
                price_data = self.search_cheapest_by_weapon(detected_weapon, limit=25)  # Increased limit for cheapest queries
                # Filter for StatTrak if specified
                if is_stattrak:
                    price_data = [r for r in price_data if "stattrak™" in r['item_name'].lower() or "stattrak" in r['item_name'].lower()]
                if price_data:
                    return price_data
            else:
                # Generic cheapest query - search all items
                all_items = []
                for item_name in self.item_names:
                    # Skip non-StatTrak items if StatTrak was specified
                    if is_stattrak and not ("stattrak™" in item_name.lower() or "stattrak" in item_name.lower()):
                        continue
                    
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
                if all_items:
                    return all_items[:25]
        
        # Case 2.5: Most expensive weapon skin query
        if "most expensive" in query or "highest price" in query or "priciest" in query:
            if detected_weapon:
                price_data = self.search_most_expensive_by_weapon(detected_weapon, limit=25)  # Return top 25 most expensive
                # Filter for StatTrak if specified
                if is_stattrak:
                    price_data = [r for r in price_data if "stattrak™" in r['item_name'].lower() or "stattrak" in r['item_name'].lower()]
                if price_data:
                    return price_data
            else:
                # Generic most expensive query - search all items
                all_items = []
                for item_name in self.item_names:
                    # Skip non-StatTrak items if StatTrak was specified
                    if is_stattrak and not ("stattrak™" in item_name.lower() or "stattrak" in item_name.lower()):
                        continue
                    
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
                
                # Sort by price (descending) and return top 25
                all_items.sort(key=lambda x: x['min_price'], reverse=True)
                if all_items:
                    return all_items[:25]
                
        # Case 3: Specific weapon + skin query
        if detected_weapon and skin_name:
            matches = self.search_by_weapon_and_skin(detected_weapon, skin_name)
            
            # Filter for StatTrak if specified
            if is_stattrak:
                stattrak_matches = [m for m in matches if "stattrak™" in m.lower() or "stattrak" in m.lower()]
                # If we found any StatTrak matches, use those; otherwise fall back to all matches
                if stattrak_matches:
                    matches = stattrak_matches
            
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
                
                if results:
                    return results[:limit] if limit else results
        
        # Case 4: Weapon-only query (no specific skin)
        if detected_weapon and not skin_name:
            # For weapon-only queries, if it's a price query, sort by price
            if is_price_query:
                price_data = self.search_cheapest_by_weapon(detected_weapon, limit=limit)
                # Filter for StatTrak if specified
                if is_stattrak and price_data:
                    price_data = [r for r in price_data if "stattrak™" in r['item_name'].lower() or "stattrak" in r['item_name'].lower()]
                if price_data:
                    return price_data
            
            # Otherwise, search all skins for this weapon
            matches = self.weapon_type_index.get(detected_weapon.lower(), [])
            
            # Filter for StatTrak if specified
            if is_stattrak and matches:
                stattrak_matches = [m for m in matches if "stattrak™" in m.lower() or "stattrak" in m.lower()]
                # If we found any StatTrak matches, use those; otherwise fall back to all matches
                if stattrak_matches:
                    matches = stattrak_matches
            
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
                
                if results:
                    return results[:limit] if limit else results
        
        # Case 5: Try exact match
        exact_matches = self.exact_match(query)
        if exact_matches:
            # Filter for StatTrak if specified
            if is_stattrak:
                stattrak_matches = [m for m in exact_matches if "stattrak™" in m.lower() or "stattrak" in m.lower()]
                # If we found any StatTrak matches, use those; otherwise fall back to all matches
                if stattrak_matches:
                    exact_matches = stattrak_matches
                
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
            
            if results:
                return results[:limit] if limit else results
            
        # Case 6: Fall back to fuzzy search
        fuzzy_results = self.fuzzy_search(query, top_k=limit if limit else 20)
        
        if fuzzy_results:
            # Filter for StatTrak if specified
            if is_stattrak:
                stattrak_fuzzy = [(name, score) for name, score in fuzzy_results 
                                 if "stattrak™" in name.lower() or "stattrak" in name.lower()]
                # If we found any StatTrak matches, use those; otherwise fall back to all matches
                if stattrak_fuzzy:
                    fuzzy_results = stattrak_fuzzy
                
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
        query_lower = query.lower()
        
        # Detect query type
        is_price_query, max_price, min_price = self.detect_price_query(query)
        
        # Check for StatTrak keyword with more variations
        is_stattrak = any(term in query_lower for term in ["stattrak™", "stattrak", "stat trak", "stat-trak", "stattrack", "st"])
        
        # Check for wear conditions
        wear_conditions = {
            "factory new": ["factory new", "fn"],
            "minimal wear": ["minimal wear", "mw"],
            "field-tested": ["field-tested", "field tested", "ft"],
            "well-worn": ["well-worn", "well worn", "ww"],
            "battle-scarred": ["battle-scarred", "battle scarred", "bs"]
        }
        
        detected_wear = None
        for wear, aliases in wear_conditions.items():
            if any(alias in query_lower for alias in aliases):
                detected_wear = wear
                break
        
        # Extract skin name if present
        skin_patterns = [
            "autotronic", "lore", "doppler", "gamma doppler", "marble fade", "tiger tooth", 
            "fade", "crimson web", "slaughter", "case hardened", "ultraviolet", "night", 
            "blue steel", "damascus steel", "rust coat", "scorched", "forest ddpat", 
            "urban masked", "stained", "safari mesh", "boreal forest"
        ]
        
        detected_skin = None
        for pattern in skin_patterns:
            if pattern in query_lower:
                detected_skin = pattern
                break
        
        # Determine if this was a specific weapon query
        weapon_names = ["ak-47", "m4a4", "m4a1-s", "awp", "desert eagle", "glock-18", "usp-s", "p250", "knife", "karambit"]
        detected_weapon = None
        for weapon in weapon_names:
            if weapon in query_lower or weapon.replace("-", "") in query_lower:
                detected_weapon = weapon.upper()
                break
        
        # If we have no results but detected specific weapon, skin, wear and StatTrak status,
        # try to find similar items that might be available
        if not results and detected_weapon and detected_skin and detected_wear and is_stattrak:
            alternate_results = []
            
            # First try: Same weapon + skin but different wear conditions
            for wear in wear_conditions.keys():
                if wear != detected_wear:
                    alt_query = f"stattrak {detected_weapon} {detected_skin} {wear}"
                    wear_specific_results = self.search(alt_query, limit=3)
                    if wear_specific_results:
                        alternate_results.extend(wear_specific_results)
            
            # If no alternatives with different wear, try non-StatTrak version
            if not alternate_results:
                non_st_query = f"{detected_weapon} {detected_skin} {detected_wear}"
                non_st_results = self.search(non_st_query, limit=3)
                if non_st_results:
                    alternate_results.extend(non_st_results)
                
            # If we found alternatives, return those with a note
            if alternate_results:
                results = alternate_results[:3]  # Limit to top 3 alternatives
                note = f"I couldn't find a StatTrak™ {detected_weapon} | {detected_skin} ({detected_wear}), but here are some related alternatives:"
                
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
                
                # Add a tip for real-time data
                tip = "\n\nNote: Prices and availability change frequently. For the most up-to-date information, check Skinport directly."
                
                return f"{note}\n\n" + "\n\n".join(formatted_results) + tip
            
        # If no results and no alternatives found
        if not results:
            base_message = f"I couldn't find any CS2 skins matching '{query}'."
            
            # Add more helpful details if we detected specific criteria
            if detected_weapon and detected_skin and detected_wear and is_stattrak:
                message = (f"I couldn't find the StatTrak™ {detected_weapon} | {detected_skin} ({detected_wear}) in the current data. "
                          f"This item might be unavailable on Skinport or our data may need to be updated. "
                          f"For real-time availability, check Skinport directly.")
                return message
            else:
                return f"{base_message} Please try using a more specific name or check your spelling."
        
        # Check if results might have been limited
        is_limited = (is_price_query and (max_price is not None or min_price is not None) and len(results) == 15)
        
        # Format the header based on query type
        header = f"Found {len(results)} CS2 skin"
        if len(results) != 1:
            header += "s"
        
        # Add StatTrak info if detected
        if is_stattrak:
            header += " (StatTrak™)"
        
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
        
        # Check if this is a "most expensive" query
        is_most_expensive_query = "most expensive" in query_lower or "highest price" in query_lower or "priciest" in query_lower
        
        # Add price-related item summary
        if len(results) > 0:
            if is_most_expensive_query:
                # For most expensive queries, highlight the most expensive item (first in results, as they're sorted)
                expensive_item = results[0]
                stattrak_label = "StatTrak™ " if is_stattrak else ""
                header += f"\nThe most expensive {stattrak_label}{detected_weapon if detected_weapon else ''} skin is {expensive_item['item_name']} at ${expensive_item['min_price']:.2f}"
            elif "cheapest" in query_lower or "lowest price" in query_lower or is_price_query:
                # For cheapest queries, highlight the cheapest item
                cheapest_item = min(results, key=lambda x: x['min_price'])
                stattrak_label = "StatTrak™ " if is_stattrak else ""
                header += f"\nThe cheapest {stattrak_label}{detected_weapon if detected_weapon else ''} skin is {cheapest_item['item_name']} at ${cheapest_item['min_price']:.2f}"
        
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
        
        # Add disclaimer about price fluctuations
        footer = "\n\nNote: Prices and availability change frequently. For real-time information, check Skinport directly."
        
        return f"{header}\n\n" + "\n\n".join(formatted_results) + footer

    def _correct_spelling(self, query: str) -> str:
        """
        Correct common misspellings in search queries
        
        Args:
            query: The original search query
            
        Returns:
            Corrected query string
        """
        # Normalize StatTrak notation
        normalized_query = query.lower().strip()
        for st_term in ["stat trak", "stat-trak", "stattrack", "st"]:
            if st_term in normalized_query:
                normalized_query = normalized_query.replace(st_term, "stattrak")
        
        # Correct common misspellings of skin names
        skin_corrections = {
            "autorinic": "autotronic",
            "autronic": "autotronic",
            "autoronic": "autotronic",
            "ultrvoilet": "ultraviolet",
            "ultraviolt": "ultraviolet",
            "doplar": "doppler",
            "doplr": "doppler",
            "marbl": "marble",
            "marbel": "marble",
            "marblefade": "marble fade",
            "tigertoot": "tiger tooth",
            "tiger toot": "tiger tooth",
            "casehardened": "case hardened",
            "case-hardened": "case hardened",
            "crim web": "crimson web",
            "crimsonweb": "crimson web",
            "blu steel": "blue steel",
            "damascus": "damascus steel",
            "rust": "rust coat", 
            "gamma dopler": "gamma doppler",
            "gamma-doppler": "gamma doppler",
            "karambit": "karambit"
        }
        
        # Apply spelling corrections to the query
        for misspelling, correction in skin_corrections.items():
            if misspelling in normalized_query:
                normalized_query = normalized_query.replace(misspelling, correction)
        
        return normalized_query


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