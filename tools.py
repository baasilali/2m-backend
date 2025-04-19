from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import json
import os

# Try to import the simplified search engine
try:
    print("Importing simplified search engine...")
    from search_utils_simplified import get_skin_search_engine
    print("Successfully imported simplified search engine")
except ImportError as e:
    print(f"Error importing simplified search engine: {e}")
    # Fall back to original search engines if needed
    try:
        print("Falling back to original search engine...")
        from search_utils import get_skin_search_engine
        print("Successfully imported original search engine")
    except ImportError:
        print("ERROR: Failed to import any search engine")
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
        marketplace_path = os.path.join(current_dir, "data", "skinport_data.json")
        
        # Initialize search engine and load data
        search_engine = get_skin_search_engine(marketplace_path)
        
        # Detect query type to determine appropriate limitations
        # The enhanced price detection will handle formats like:
        # - "under $10"
        # - "between $20 and $50"
        # - "cheapest AK-47"
        # - "most expensive knife"
        
        # Let the search engine's sophisticated query detection handle limitations
        is_price_query, max_price, min_price = search_engine.detect_price_query(query)
        query_lower = query.lower()
        
        # No limit for:
        # 1. Cheapest/most expensive queries
        # 2. Price range queries (under $X, over $X)
        # 3. Specific weapon price queries
        if "cheapest" in query_lower or "most expensive" in query_lower or max_price is not None or min_price is not None:
            limit = None
        else:
            # Default limit for general queries
            limit = 15
            
        # Log search parameters for debugging
        print(f"Search query: '{query}'")
        print(f"Price query: {is_price_query}, Max: {max_price}, Min: {min_price}, Limit: {limit}")
        
        # Perform the search with our enhanced engine
        results = search_engine.search(query, limit=limit)
        
        # Format the results nicely
        return search_engine.format_search_results(results, query)
    
    except Exception as e:
        return f"An error occurred while searching for skins: {str(e)}"

cs_skins_tool = Tool(
    name="cs_skins",
    func=query_cs_skins,
    description="Retrieve detailed information about Counter Strike skin prices from Skinport marketplace data. Handles various price queries including 'cheapest AK-47', 'skins under $10', 'AWP between $50 and $100', etc. Returns comprehensive results sorted by price for price-related queries.",
)

