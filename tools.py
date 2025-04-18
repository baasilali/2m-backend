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
        marketplace_path = os.path.join(current_dir, "data", "skinport_data.json")
        
        # Initialize search engine and load data
        search_engine = get_skin_search_engine(marketplace_path)
        items = search_engine.items
        
        if not items:
            return "Marketplace data not available or empty."
        
        query = query.lower().strip()
        
        # Use semantic search to find relevant items
        item_candidates = []
        item_scores = {}
        
        # Get search results
        search_results = search_engine.hybrid_search(query, top_k=3)  # Reduced to 3 for more focused results
        
        # If no results found with semantic search, try partial match as fallback
        if not search_results:
            # Display informative message if no matches
            return f"I couldn't find any items matching '{query}'. Please try using a more specific name or check your spelling."
        
        # Extract item names and their scores
        item_candidates = [result['item_name'] for result in search_results]
        item_scores = {result['item_name']: result['total_score'] for result in search_results}
        
        # For price queries, show detailed information for the top match
        if any(term in query for term in ["price", "cost", "value", "how much", "min", "max", "average"]):
            if len(item_candidates) >= 1:
                # Get the top match
                item_name = item_candidates[0]
                item_data = items[item_name]
                
                # Create a focused summary with clear price distinctions
                summary = f"The {item_name}:\n"
                summary += f"• SkinPort suggests: ${item_data.get('suggested_price', 'N/A')}\n"
                summary += f"• Actual market prices: ${item_data.get('min_price', 'N/A')} - ${item_data.get('max_price', 'N/A')}\n"
                summary += f"• Available quantity: {item_data.get('quantity', 'N/A')} items"
                
                return summary
            else:
                return f"I couldn't find any items matching '{query}'. Please try a more specific item name."
        
        # Generic item queries - show multiple matches with detailed information
        if item_candidates:
            result = []
            for item_name in item_candidates:
                item_data = items[item_name]
                result.append(f"{item_name}:\n• Suggested: ${item_data.get('suggested_price', 'N/A')}\n• Market: ${item_data.get('min_price', 'N/A')} - ${item_data.get('max_price', 'N/A')}\n• Available: {item_data.get('quantity', 'N/A')}")
            
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

