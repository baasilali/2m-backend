from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import json
import os

# Try to import the full search engine first, fall back to simpler version if needed
try:
    print("Trying to import full search engine with embeddings...")
    from search_utils import get_skin_search_engine
    print("Successfully imported full search engine")
except ImportError as e:
    print(f"Error importing full search engine: {e}")
    print("Falling back to simplified fuzzy search engine")
    try:
        from search_utils_fallback import get_skin_search_engine
        print("Successfully imported fallback search engine")
    except ImportError:
        print("ERROR: Failed to import either search engine")
        raise

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
        
        # Initialize search engine and load data
        search_engine = get_skin_search_engine(marketplace_path)
        items = search_engine.items
        
        if not items:
            return "Marketplace data not available or empty."
        
        query = query.lower().strip()
        
        # First, try to extract marketplace from query if mentioned
        market_name = None
        for market in ["skinport", "dmarket", "csfloat", "buff163"]:
            if market in query:
                if market == "skinport":
                    market_name = "SkinPort"
                elif market == "dmarket":
                    market_name = "DMarket"
                elif market == "csfloat":
                    market_name = "CSFloat"
                elif market == "buff" or market == "buff163":
                    market_name = "Buff163"
                break
        
        # Use semantic search to find relevant items
        item_candidates = []
        item_scores = {}
        
        # If specific market mentioned, filter query to focus on item names
        search_query = query
        if market_name:
            search_query = query.replace(market_name.lower(), "").strip()
        
        # Get search results
        search_results = search_engine.hybrid_search(search_query, top_k=5)
        
        # If no results found with semantic search, try partial match as fallback
        if not search_results:
            # Display informative message if no matches
            return f"I couldn't find any items matching '{query}'. Please try using a more specific name or check your spelling."
        
        # Extract item names and their scores
        item_candidates = [result['item_name'] for result in search_results]
        item_scores = {result['item_name']: result['total_score'] for result in search_results}
        
        # Handle marketplace-specific queries
        if market_name:
            market_results = []
            for item_name in item_candidates:
                market_data = items[item_name].get(market_name, {})
                if market_data:
                    price = market_data.get("price", "N/A")
                    score = item_scores.get(item_name, 0)
                    # Format the price based on marketplace
                    if price is None:
                        price_str = "Not available"
                    elif market_name == "DMarket" and isinstance(price, str):
                        try:
                            price = float(price) / 100
                            price_str = f"${price:.2f}"  # DMarket prices in cents
                        except:
                            price_str = f"${price}"
                    elif market_name == "Buff163" and isinstance(price, str):
                        try:
                            usd_price = float(price) / 7
                            price_str = f"¥{price} (~${usd_price:.2f})"  # Rough USD conversion
                        except:
                            price_str = f"¥{price}"
                    else:
                        if isinstance(price, (int, float)):
                            price_str = f"${price:.2f}"
                        else:
                            price_str = f"${price}"
                    
                    market_results.append((item_name, price_str, score))
            
            if market_results:
                # Sort by score (descending)
                market_results.sort(key=lambda x: x[2], reverse=True)
                result_lines = [f"Found items matching '{search_query}' on {market_name}:"]
                for item_name, price_str, _ in market_results:
                    result_lines.append(f"- {item_name}: {price_str}")
                return "\n".join(result_lines)
            else:
                # List available items as a fallback
                available_items = []
                for item_name, markets in items.items():
                    if market_name in markets:
                        available_items.append(item_name)
                
                if available_items:
                    sample = available_items[:10]
                    return f"No items matching '{search_query}' found on {market_name}. Here are some available items:\n" + "\n".join([f"- {name}" for name in sample]) + f"\n\n({len(available_items)} total items available)"
                else:
                    return f"No items found on {market_name}."
        
        # For price queries, show detailed information for the top match
        if "price" in query or "cost" in query or "value" in query or "how much" in query:
            if len(item_candidates) >= 1:
                # Get the top match
                item_name = item_candidates[0]
                result = [f"Marketplace prices for {item_name}:"]
                
                for market_name, data in items[item_name].items():
                    price = data.get("price", "N/A")
                    
                    # Handle different price formats
                    if price is None:
                        price_str = "Not available"
                    elif market_name == "DMarket" and isinstance(price, str):
                        try:
                            price_str = f"${float(price) / 100:.2f}"  # DMarket prices in cents
                        except:
                            price_str = f"${price}"
                    elif market_name == "Buff163" and isinstance(price, str):
                        try:
                            usd_price = float(price) / 7
                            price_str = f"¥{price} (~${usd_price:.2f})"  # Rough USD conversion
                        except:
                            price_str = f"¥{price}"
                    else:
                        if isinstance(price, (int, float)):
                            price_str = f"${price:.2f}"
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
                
                # For multiple matches, show alternatives
                if len(item_candidates) > 1:
                    result.append("\nDid you mean one of these instead?")
                    for i, name in enumerate(item_candidates[1:4], 1):  # Show next 3 alternatives
                        result.append(f"- {name}")
                
                return "\n".join(result)
            else:
                return f"I couldn't find any items matching '{query}'. Please try a more specific item name."
        
        # Generic item queries - show multiple matches with summary information
        if item_candidates:
            result = [f"Found items matching '{query}':"]
            for item_name in item_candidates[:5]:  # Limit to top 5 matches
                # Calculate average price across marketplaces
                avg_price = 0
                count = 0
                available_markets = []
                
                for market, data in items[item_name].items():
                    price = data.get("price", 0)
                    if price is None:
                        continue
                    
                    if market == "DMarket" and isinstance(price, str):
                        try:
                            price = float(price) / 100  # DMarket prices in cents
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
                        available_markets.append(market)
                
                if count > 0:
                    avg_price = avg_price / count
                    markets_str = ", ".join(available_markets)
                    result.append(f"- {item_name}: ~${avg_price:.2f} (avg across {markets_str})")
                else:
                    result.append(f"- {item_name}: Price data unavailable")
            
            return "\n".join(result)
        
        # No items found with any method
        return f"I couldn't find any items matching '{query}'. Please try using a more specific name or check your spelling."
    
    except Exception as e:
        return f"An error occurred while searching for skins: {str(e)}"

cs_skins_tool = Tool(
    name="cs_skins",
    func=query_cs_skins,
    description="Retrieve information about Counter Strike skin prices from marketplace data including SkinPort, DMarket and CSFloat.",
)

