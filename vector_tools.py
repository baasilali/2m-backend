from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import json
import os
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the enhanced systems
try:
    from vector_search_engine import get_vector_skin_search_engine
    has_vector_search = True
    logger.info("Successfully imported vector search engine")
except ImportError as e:
    has_vector_search = False
    logger.error(f"Error importing vector search engine: {e}")

try:
    from enhanced_document_tools import enhanced_document_tool
    has_enhanced_docs = True
    logger.info("Successfully imported enhanced document tools")
except ImportError as e:
    has_enhanced_docs = False
    logger.error(f"Error importing enhanced document tools: {e}")

def save_to_txt(data: str, filename: str = "research_output.txt"):
    """Save research data to text file"""
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

# Web search tools (unchanged)
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

def query_cs_skins_vector(query: str) -> str:
    """
    Advanced CS2 skin search using vector embeddings for semantic understanding.
    This provides much more accurate results than basic fuzzy matching.
    """
    try:
        if not has_vector_search:
            return "Vector search engine not available. Please install required dependencies."
        
        logger.info(f"Vector search query: '{query}'")
        
        # Initialize the vector search engine
        search_engine = get_vector_skin_search_engine()
        
        # Check data freshness
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "skinport_data.json")
        refresh_message = ""
        
        if os.path.exists(data_path):
            mod_time = os.path.getmtime(data_path)
            current_time = time.time()
            hours_since_update = (current_time - mod_time) / 3600
            
            if hours_since_update > 24:
                days_old = int(hours_since_update / 24)
                refresh_message = f"\n\n⚠️ **Data Freshness Notice**: Price data is {days_old} day{'s' if days_old > 1 else ''} old. Some items or prices may have changed."
        
        # Perform vector-based search
        results = search_engine.search(query, limit=10)
        
        if not results:
            return f"No CS2 skins found matching '{query}' using semantic search. Try different keywords or more general terms."
        
        # Classify query intent for better formatting
        intent = search_engine.classify_intent(query)
        
        # Format results with enhanced context
        output = []
        output.append(f"🎯 **Vector Search Results for '{query}'** ({len(results)} items found)\n")
        
        # Show high-relevance results first
        high_relevance = [r for r in results if r.get('relevance_score', 0) > 0.8]
        if high_relevance and len(results) > 3:
            output.append("✨ **Highly Relevant Matches:**")
            for i, item in enumerate(high_relevance[:3], 1):
                output.append(f"{i}. {_format_vector_item(item, intent)}")
            output.append("")
        
        # Show all results
        if not high_relevance or len(high_relevance) < len(results):
            if high_relevance:
                output.append("📋 **All Results:**")
            
            for i, item in enumerate(results, 1):
                output.append(f"{i}. {_format_vector_item(item, intent)}")
        
        # Add search insights
        if intent.get('price_range'):
            price_items = [float(item.get('suggested_price', 0)) for item in results if item.get('suggested_price')]
            if price_items:
                min_price = min(price_items)
                max_price = max(price_items)
                avg_price = sum(price_items) / len(price_items)
                output.append(f"\n💰 **Price Range**: ${min_price:.2f} - ${max_price:.2f} (Average: ${avg_price:.2f})")
        
        # Add semantic search explanation
        output.append(f"\n🔍 **Search powered by vector embeddings** - Results ranked by semantic similarity to understand intent and context.")
        
        # Add data freshness warning if needed
        if refresh_message:
            output.append(refresh_message)
        
        return "\n".join(output)
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in vector CS2 skin search: {error_msg}")
        return f"An error occurred while performing vector search for CS2 skins: {error_msg}. Please try a more specific query."

def _format_vector_item(item: dict, intent: dict) -> str:
    """Format a single item from vector search results"""
    name = item.get('item_name', 'Unknown')
    price = item.get('suggested_price', 'N/A')
    relevance = item.get('relevance_score', 0)
    
    try:
        price_val = float(price)
        price_str = f"${price_val:.2f}"
    except (ValueError, TypeError):
        price_str = "Price N/A"
    
    # Add relevance indicator for high-confidence matches
    relevance_indicator = ""
    if relevance > 0.9:
        relevance_indicator = " 🎯"
    elif relevance > 0.8:
        relevance_indicator = " ✨"
    
    return f"**{name}** - {price_str}{relevance_indicator} (Relevance: {relevance:.2f})"

# Create the vector-based CS2 skins tool
cs_skins_vector_tool = Tool(
    name="cs_skins_vector",
    func=query_cs_skins_vector,
    description="Advanced semantic search for CS2 skins using vector embeddings. Understands intent and context much better than basic text matching. Handles complex queries like 'affordable AK-47 skins with good patterns', 'best investment knife under $200', or 'popular rifle skins for competitive play'. Provides relevance-ranked results with semantic understanding.",
)

# Enhanced document tool (if available)
if has_enhanced_docs:
    vector_document_tool = enhanced_document_tool
else:
    # Fallback to basic document tool
    from document_tools import document_tool
    vector_document_tool = document_tool

# Export tools for use in main application
vector_tools = [cs_skins_vector_tool, vector_document_tool, search_tool, wiki_tool, save_tool]

# Compatibility function to get tools
def get_vector_tools():
    """Get the vector-enhanced tools"""
    return vector_tools 