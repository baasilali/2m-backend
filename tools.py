from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import json
import os
import time

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
        # Import search engine utilities
        from search_utils_simplified import get_skin_search_engine
        import os
        import time
        import json
        from datetime import datetime, timedelta
        
        # Initialize the search engine
        search_engine = get_skin_search_engine()
        
        # Check data freshness
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "skinport_data.json")
        data_is_stale = False
        refresh_message = ""
        
        if os.path.exists(data_path):
            # Get file modification time
            mod_time = os.path.getmtime(data_path)
            current_time = time.time()
            
            # Calculate hours since last update
            hours_since_update = (current_time - mod_time) / 3600
            
            # If data is older than 24 hours, consider it stale
            if hours_since_update > 24:
                data_is_stale = True
                days_old = int(hours_since_update / 24)
                refresh_message = f"\n\nNote: Price data is {days_old} day{'s' if days_old > 1 else ''} old. Some items or prices may have changed."
        
        # Parse the query for price thresholds
        is_price_query, max_price, min_price = search_engine.detect_price_query(query)
        
        # Set limit based on query type (more results for price queries)
        limit = 15 if is_price_query else 10
        
        # Log the search query for analytics and debugging
        print(f"Search query: '{query}'")
        print(f"Price query: {is_price_query}, Max: {max_price}, Min: {min_price}, Limit: {limit}")
        
        # Choose search method based on query type
        if is_price_query:
            # For price queries, use the regular search which has specific handling for price ranges
            results = search_engine.search(query, limit=limit)
        else:
            # For non-price queries, use the hierarchical search for better accuracy
            results = search_engine.hierarchical_search(query, limit=limit)
            # If no results, fall back to regular search
            if not results:
                print("No results from hierarchical search, trying regular search")
                results = search_engine.search(query, limit=limit)
        
        # Format the results nicely
        formatted_results = search_engine.format_search_results(results, query)
        
        # Add the data freshness warning if needed
        if data_is_stale:
            formatted_results += refresh_message
            
        return formatted_results
    
    except Exception as e:
        error_msg = str(e)
        print(f"Error in CS2 skin search: {error_msg}")
        return f"An error occurred while searching for CS2 skins: {error_msg}. Please try a more specific query or check your spelling."

cs_skins_tool = Tool(
    name="cs_skins",
    func=query_cs_skins,
    description="Retrieve detailed information about Counter Strike skin prices from Skinport marketplace data. Handles various price queries including 'cheapest AK-47', 'skins under $10', 'AWP between $50 and $100', etc. Returns comprehensive results sorted by price for price-related queries.",
)

