from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import json
import os

def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

def query_cs_skins(query: str) -> str:
    """Query the Counter Strike marketplace skin database."""
    try:
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        marketplace_path = os.path.join(current_dir, "data", "prices_output.json")
        
        # Load the marketplace data JSON
        with open(marketplace_path, 'r') as file:
            marketplace_data = json.load(file)
        
        # Check if the data has the expected structure
        # If there's a "marketplace_data" key, use that
        # Otherwise, assume the skins are at the top level
        if "marketplace_data" in marketplace_data:
            items = marketplace_data.get("marketplace_data", {})
        else:
            # Assume the skins are at the top level
            items = marketplace_data
        
        if not items:
            return "Marketplace data not available or empty."
        
        query = query.lower().strip()
        
        # Check for specific marketplace platform queries
        if any(market in query for market in ["skinport", "dmarket", "csfloat", "buff163"]):
            market_name = ""
            if "skinport" in query:
                market_name = "SkinPort"
            elif "dmarket" in query:
                market_name = "DMarket" 
            elif "csfloat" in query:
                market_name = "CSFloat"
            elif "buff" in query or "buff163" in query:
                market_name = "Buff163"
            
            # Extract item name from query
            found_items = []
            for item_name in items:
                if any(term in item_name.lower() for term in query.replace(market_name.lower(), "").split()):
                    market_data = items[item_name].get(market_name, {})
                    if market_data:
                        price = market_data.get("price", "N/A")
                        # Handle different price formats
                        if price is None:
                            price = "Not available"
                        elif market_name == "DMarket" and isinstance(price, str):
                            try:
                                price = float(price) / 100  # DMarket prices appear to be in cents
                            except:
                                pass
                        found_items.append(f"{item_name} on {market_name}: ${price}")
            
            if found_items:
                return "\n".join(found_items)
            else:
                # List all available items on that marketplace
                available_items = []
                for item_name, markets in items.items():
                    if market_name in markets:
                        available_items.append(item_name)
                
                # Limit results to avoid overwhelming output
                if available_items:
                    sample = available_items[:10]
                    return f"No specific items found on {market_name}. Here are some available items:\n" + "\n".join([f"- {name}" for name in sample]) + f"\n\n({len(available_items)} total items available)"
                else:
                    return f"No items found on {market_name}."
        
        # Handle exact name matches
        for item_name in items:
            # Check for exact name match
            if query.lower() in item_name.lower():
                result = [f"Marketplace prices for {item_name}:"]
                
                for market_name, data in items[item_name].items():
                    price = data.get("price", "N/A")
                    
                    # Handle different price formats
                    if price is None:
                        price_str = "Not available"
                    elif market_name == "DMarket" and isinstance(price, str):
                        try:
                            price_str = f"${float(price) / 100}"  # DMarket prices appear to be in cents
                        except:
                            price_str = f"${price}"
                    elif market_name == "Buff163" and isinstance(price, str):
                        try:
                            price_str = f"¥{price} (~${float(price) / 7})"  # Rough USD conversion
                        except:
                            price_str = f"¥{price}"
                    else:
                        price_str = f"${price}"
                    
                    result.append(f"- {market_name}: {price_str}")
                    
                    # Add additional details if available
                    if "float" in data:
                        result.append(f"  Float value: {data['float']}")
                    elif "floatValue" in data:
                        result.append(f"  Float value: {data['floatValue']}")
                    
                    if "quality" in data:
                        result.append(f"  Quality: {data['quality']}")
                    
                    if "stattrak" in data and data["stattrak"]:
                        result.append(f"  StatTrak™: Yes")
                    
                    if "souvenir" in data and data["souvenir"]:
                        result.append(f"  Souvenir: Yes")
                
                return "\n".join(result)
        
        # Handle partial matches for skins
        if "price" in query or "cost" in query or "value" in query:
            matching_items = []
            for item_name in items:
                match_score = 0
                for term in query.split():
                    if term in ["price", "cost", "value", "how", "much", "is"]:
                        continue
                    if term in item_name.lower():
                        match_score += 1
                
                if match_score >= 1:  # At least one matching term
                    matching_items.append(item_name)
            
            if matching_items:
                if len(matching_items) == 1:
                    # If only one match, show full details
                    item_name = matching_items[0]
                    result = [f"Marketplace prices for {item_name}:"]
                    
                    for market_name, data in items[item_name].items():
                        price = data.get("price", "N/A")
                        if price is None:
                            price_str = "Not available"
                        elif market_name == "DMarket" and isinstance(price, str):
                            try:
                                price_str = f"${float(price) / 100}"
                            except:
                                price_str = f"${price}"
                        elif market_name == "Buff163" and isinstance(price, str):
                            try:
                                price_str = f"¥{price} (~${float(price) / 7})"
                            except:
                                price_str = f"¥{price}"
                        else:
                            price_str = f"${price}"
                        
                        result.append(f"- {market_name}: {price_str}")
                    
                    return "\n".join(result)
                else:
                    # If multiple matches, show a summary
                    result = ["Found multiple matching items:"]
                    for item_name in matching_items[:10]:  # Limit to 10 items
                        avg_price = 0
                        count = 0
                        for market, data in items[item_name].items():
                            price = data.get("price", 0)
                            if price is None:
                                continue
                            
                            if market == "DMarket" and isinstance(price, str):
                                try:
                                    price = float(price) / 100
                                except:
                                    price = 0
                            elif market == "Buff163" and isinstance(price, str):
                                try:
                                    price = float(price) / 7  # Rough USD conversion
                                except:
                                    price = 0
                            
                            if price:
                                avg_price += float(price)
                                count += 1
                        
                        if count > 0:
                            avg_price = avg_price / count
                            result.append(f"- {item_name}: ~${avg_price:.2f} (avg)")
                        else:
                            result.append(f"- {item_name}: Price data unavailable")
                    
                    if len(matching_items) > 10:
                        result.append(f"\nAnd {len(matching_items) - 10} more matching items...")
                    
                    return "\n".join(result)
        
        # Handle item category queries
        if "knife" in query or "knives" in query:
            knife_skins = [name for name in items.keys() if "knife" in name.lower()]
            if not knife_skins:
                return "No knife skins found in the database."
            
            result = ["Available knife skins:"]
            for name in knife_skins[:20]:  # Limit to 20 to avoid overwhelming output
                result.append(f"- {name}")
            
            if len(knife_skins) > 20:
                result.append(f"\nAnd {len(knife_skins) - 20} more knife skins...")
            
            return "\n".join(result)
        
        if "gloves" in query or "glove" in query:
            glove_skins = [name for name in items.keys() if "gloves" in name.lower()]
            if not glove_skins:
                return "No gloves found in the database."
            
            result = ["Available gloves:"]
            for name in glove_skins[:20]:
                result.append(f"- {name}")
            
            if len(glove_skins) > 20:
                result.append(f"\nAnd {len(glove_skins) - 20} more gloves...")
            
            return "\n".join(result)
        
        # Handle weapon-specific queries
        weapons = ["ak-47", "m4a4", "m4a1-s", "awp", "desert eagle", "deagle", "usp-s", "glock-18", "p250", "p90"]
        for weapon in weapons:
            if weapon in query:
                weapon_skins = [name for name in items.keys() if weapon.lower() in name.lower()]
                if not weapon_skins:
                    return f"No {weapon.upper()} skins found in the database."
                
                result = [f"Available {weapon.upper()} skins:"]
                for name in weapon_skins[:20]:
                    result.append(f"- {name}")
                
                if len(weapon_skins) > 20:
                    result.append(f"\nAnd {len(weapon_skins) - 20} more {weapon.upper()} skins...")
                
                return "\n".join(result)
        
        # Handle StatTrak queries
        if "stattrak" in query or "stat trak" in query or "stat-trak" in query:
            stattrak_items = []
            for item_name, markets in items.items():
                for market, data in markets.items():
                    if "stattrak" in data and data["stattrak"]:
                        stattrak_items.append(item_name)
                        break
            
            if not stattrak_items:
                return "No StatTrak™ items found in the database."
            
            result = ["Available StatTrak™ items:"]
            for name in stattrak_items[:20]:
                result.append(f"- {name}")
            
            if len(stattrak_items) > 20:
                result.append(f"\nAnd {len(stattrak_items) - 20} more StatTrak™ items...")
            
            return "\n".join(result)
        
        # Handle Souvenir queries
        if "souvenir" in query:
            souvenir_items = []
            for item_name, markets in items.items():
                for market, data in markets.items():
                    if "souvenir" in data and data["souvenir"]:
                        souvenir_items.append(item_name)
                        break
            
            if not souvenir_items:
                return "No Souvenir items found in the database."
            
            result = ["Available Souvenir items:"]
            for name in souvenir_items[:20]:
                result.append(f"- {name}")
            
            if len(souvenir_items) > 20:
                result.append(f"\nAnd {len(souvenir_items) - 20} more Souvenir items...")
            
            return "\n".join(result)
        
        # Handle price range queries
        if "expensive" in query or "costly" in query or "high value" in query:
            # Find the most expensive items
            item_avg_prices = []
            for item_name, markets in items.items():
                prices = []
                for market, data in markets.items():
                    price = data.get("price", 0)
                    if price is None:
                        continue
                        
                    if market == "DMarket" and isinstance(price, str):
                        try:
                            price = float(price) / 100
                        except:
                            price = 0
                    elif market == "Buff163" and isinstance(price, str):
                        try:
                            price = float(price) / 7
                        except:
                            price = 0
                            
                    if price:
                        prices.append(float(price))
                
                if prices:
                    avg_price = sum(prices) / len(prices)
                    item_avg_prices.append((item_name, avg_price))
            
            # Sort by price (highest first)
            item_avg_prices.sort(key=lambda x: x[1], reverse=True)
            
            result = ["Most expensive CS:GO skins:"]
            for name, price in item_avg_prices[:20]:
                result.append(f"- {name}: ${price:.2f} (avg)")
            
            return "\n".join(result)
        
        if "cheap" in query or "affordable" in query or "low cost" in query or "budget" in query:
            # Find the cheapest items (exclude items with price 0)
            item_avg_prices = []
            for item_name, markets in items.items():
                prices = []
                for market, data in markets.items():
                    price = data.get("price", 0)
                    if price is None:
                        continue
                        
                    if market == "DMarket" and isinstance(price, str):
                        try:
                            price = float(price) / 100
                        except:
                            price = 0
                    elif market == "Buff163" and isinstance(price, str):
                        try:
                            price = float(price) / 7
                        except:
                            price = 0
                            
                    if price > 0:  # Exclude free items or items with no price
                        prices.append(float(price))
                
                if prices:
                    avg_price = sum(prices) / len(prices)
                    item_avg_prices.append((item_name, avg_price))
            
            # Sort by price (lowest first)
            item_avg_prices.sort(key=lambda x: x[1])
            
            result = ["Most affordable CS:GO skins:"]
            for name, price in item_avg_prices[:20]:
                result.append(f"- {name}: ${price:.2f} (avg)")
            
            return "\n".join(result)
        
        # Compare marketplaces
        if "compare" in query and "marketplace" in query:
            result = ["Marketplace Comparison (based on available data):"]
            
            # Calculate average prices across marketplaces
            marketplace_prices = {"SkinPort": [], "DMarket": [], "CSFloat": [], "Buff163": []}
            for item_name, markets in items.items():
                for market, data in markets.items():
                    price = data.get("price", 0)
                    if price is None:
                        continue
                        
                    if market == "DMarket" and isinstance(price, str):
                        try:
                            price = float(price) / 100
                        except:
                            price = 0
                    elif market == "Buff163" and isinstance(price, str):
                        try:
                            price = float(price) / 7
                        except:
                            price = 0
                            
                    if price > 0:
                        marketplace_prices[market].append(float(price))
            
            # Calculate stats
            for market, prices in marketplace_prices.items():
                if prices:
                    avg_price = sum(prices) / len(prices)
                    result.append(f"- {market}:")
                    result.append(f"  • Items listed: {len(prices)}")
                    result.append(f"  • Average price: ${avg_price:.2f}")
                    result.append(f"  • Lowest price: ${min(prices):.2f}")
                    result.append(f"  • Highest price: ${max(prices):.2f}")
            
            return "\n".join(result)
        
        # Default response with overview
        sample_items = list(items.keys())[:5]  # Get first 5 items as samples
        total_items = len(items)
        
        # Identify marketplaces in the data
        marketplaces = set()
        for item in items.values():
            marketplaces.update(item.keys())
        
        return (f"CS:GO Skin Marketplace Database\n"
                f"Contains data for {total_items} items across {', '.join(marketplaces)}.\n\n"
                f"Sample items:\n" + "\n".join([f"- {item}" for item in sample_items]) + 
                f"\n\nTry queries like:\n"
                f"- 'AWP Dragon Lore price'\n"
                f"- 'Karambit Doppler on SkinPort'\n"
                f"- 'Show me all knives'\n"
                f"- 'Most expensive skins'\n"
                f"- 'Compare marketplaces'\n"
                f"- 'StatTrak items'\n"
                f"- 'All AK-47 skins'")
            
    except Exception as e:
        return f"Error querying skin data: {str(e)}"

cs_skins_tool = Tool(
    name="cs_skins",
    func=query_cs_skins,
    description="Retrieve information about Counter Strike skin prices from marketplace data including SkinPort, DMarket and CSFloat.",
)

