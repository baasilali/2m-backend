from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import AgentExecutor, create_openai_functions_agent
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import vector-enhanced tools first, fallback to original tools
try:
    from vector_tools import get_vector_tools
    tools = get_vector_tools()
    using_vector_search = True
    logger.info("Successfully loaded vector-enhanced tools")
except ImportError as e:
    logger.warning(f"Could not import vector tools: {e}")
    logger.info("Falling back to original tools")
    from tools import search_tool, wiki_tool, save_tool, cs_skins_tool
    from document_tools import document_tool
    tools = [cs_skins_tool, search_tool, wiki_tool, save_tool, document_tool]
    using_vector_search = False

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize any search engines at startup
@app.on_event("startup")
async def startup_event():
    if using_vector_search:
        try:
            logger.info("Initializing vector search engine...")
            from vector_search_engine import get_vector_skin_search_engine
            search_engine = get_vector_skin_search_engine()
            logger.info(f"Vector search engine initialized with {len(search_engine.item_names)} items")
        except Exception as e:
            logger.error(f"Error initializing vector search engine: {str(e)}")
    else:
        try:
            logger.info("Initializing fallback search engine...")
            from search_utils_simplified import get_skin_search_engine
            search_engine = get_skin_search_engine()
            logger.info(f"Fallback search engine initialized with {len(search_engine.item_names)} items")
        except Exception as e:
            logger.error(f"Error initializing search engine: {str(e)}")

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str] = []
    search_method: str = "unknown"

llm = ChatOpenAI(model="gpt-4o-mini")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Enhanced system prompt that takes advantage of vector search capabilities
system_prompt = """
You are an advanced CS2 (Counter-Strike 2) skin economy and marketplace research assistant.
Your primary focus is on CS2 skins, trading, market analysis, and price information.
Provide detailed, accurate, and comprehensive responses with concrete examples and data.

SEARCH CAPABILITIES:
""" + ("""
🚀 **VECTOR SEARCH ENABLED** - You now have access to advanced semantic search capabilities:
- cs_skins_vector tool - Uses AI-powered vector embeddings for semantic understanding of skin queries
- query_enhanced_documents tool - Advanced document search with context awareness
- search tool - Web search for recent trends and news
- wiki_tool - General CS2 background information

Vector Search Advantages:
- Understands intent: "affordable AK skins" vs "cheap AK skins" vs "budget AK options"
- Context awareness: "best investment skins" finds valuable items even without price mentions
- Pattern recognition: "fade skins" finds all fade patterns across weapons
- Semantic matching: "sniper rifles" matches AWP without exact keyword matching
""" if using_vector_search else """
⚡ **BASIC SEARCH MODE** - Using fallback text-based search:
- cs_skins tool - Basic fuzzy matching for skin names and prices
- query_documents tool - Basic document search
- search tool - Web search for recent trends
- wiki_tool - General CS2 information
""") + """

Tool Priority (use in this order):
1. """ + ("cs_skins_vector" if using_vector_search else "cs_skins") + """ tool - For current skin prices and marketplace data from Skinport
2. """ + ("query_enhanced_documents" if using_vector_search else "query_documents") + """ tool - For detailed CS2-specific information
3. search tool - For recent CS2 market trends and news
4. wiki_tool - ONLY as a last resort for general CS2 history/background

Enhanced Query Handling:
- For complex queries, leverage the semantic understanding of vector search
- When users ask vague questions, the vector search can infer intent
- Provide context about why specific results were chosen
- Explain the semantic relationships when relevant

Response Quality Guidelines:
- Provide SPECIFIC and DETAILED information, not vague generalizations
- When discussing prices, include exact numbers (e.g., "$45.67" not "around $40-50")
- Format responses with clear sections and bullet points when appropriate
- For vector search results, mention the relevance scores when helpful
- When comparing items, create clear side-by-side comparisons
- Always include relevant market context (e.g., rarity, popularity trends)
- """ + ("Leverage the semantic understanding to provide insights about user intent" if using_vector_search else "Use multiple search attempts if initial results are insufficient") + """

IMPORTANT:
- ALWAYS interpret queries in the context of CS2 skins and trading
- If a query seems unrelated to CS2, try to find a CS2-relevant angle
- Only use Wikipedia for general CS2 history/background information
- Never provide information about other games or unrelated topics
- For ANY pricing or marketplace information, ONLY reference Skinport data
- """ + ("Explain when vector search provides better understanding than basic text matching" if using_vector_search else "If searches return poor results, try rephrasing the query") + """

Handling Incomplete Information:
- If document tools return "INCOMPLETE_INFO", ALWAYS use the search tool for more details
- Never make up or guess information when you don't have complete data
- """ + ("Use vector search's relevance scores to assess information quality" if using_vector_search else "Try multiple search variations if results are poor") + """
- When combining information from multiple sources, clearly indicate which parts come from where

Answer the user query using the tools in priority order.
Wrap the output in this format and provide no other text\n{format_instructions}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    ("assistant", "{agent_scratchpad}")
]).partial(format_instructions=parser.get_format_instructions())

agent = create_openai_functions_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def print_response(response):
    """Pretty print the structured response"""
    print("\n" + "="*50)
    print(f"TOPIC: {response.topic}")
    print("-"*50)
    print(f"SUMMARY:\n{response.summary}")
    print("-"*50)
    print("SOURCES:")
    for source in response.sources:
        print(f"- {source}")
    print("-"*50)
    print("TOOLS USED:")
    for tool in response.tools_used:
        print(f"- {tool}")
    print("="*50 + "\n")

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """
    Enhanced endpoint that handles queries using vector search when available
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        logger.info(f"Received query: {request.query}")
        logger.info(f"Using vector search: {using_vector_search}")
        
        raw_response = agent_executor.invoke({"input": request.query})
        
        try:
            structured_response = parser.parse(raw_response.get("output", ""))
            
            # Determine search method used
            search_method = "vector_enhanced" if using_vector_search else "basic_fuzzy"
            
            return QueryResponse(
                answer=structured_response.summary,
                sources=structured_response.sources,
                search_method=search_method
            )
        except Exception as e:
            logger.error(f"Error parsing structured response: {str(e)}")
            
            # Fallback: Return the raw output
            return QueryResponse(
                answer=raw_response.get("output", "Sorry, I couldn't process that request."),
                sources=[],
                search_method="fallback"
            )
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/")
def read_root():
    search_status = "Vector-Enhanced" if using_vector_search else "Basic"
    return {
        "message": f"CS2 Skin Economy API is running with {search_status} search!",
        "vector_search_enabled": using_vector_search,
        "available_tools": [tool.name for tool in tools]
    }

@app.get("/search-info")
def get_search_info():
    """Endpoint to get information about the current search capabilities"""
    return {
        "vector_search_enabled": using_vector_search,
        "search_method": "vector_enhanced" if using_vector_search else "basic_fuzzy",
        "capabilities": {
            "semantic_understanding": using_vector_search,
            "intent_classification": using_vector_search,
            "context_awareness": using_vector_search,
            "relevance_scoring": using_vector_search
        },
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "enhanced": "vector" in tool.name.lower() or "enhanced" in tool.name.lower()
            }
            for tool in tools
        ]
    }

def main():
    search_type = "Vector-Enhanced" if using_vector_search else "Basic"
    print(f"CS2 Skin Economy Research Assistant - {search_type} Mode")
    print("Type 'exit', 'quit', or 'q' to end the session")
    
    if using_vector_search:
        print("\n🚀 Vector search enabled! Try complex queries like:")
        print("- 'affordable AK-47 skins with good patterns'")
        print("- 'best investment knives under $200'")
        print("- 'popular rifle skins for competitive play'")
    else:
        print("\n⚡ Using basic search mode. For better results, install vector dependencies.")
    
    while True:
        query = input("\nWhat can I help you with? ")
        
        # Check for exit command
        if query.lower() in ['exit', 'quit', 'q']:
            print("Thank you for using the CS2 Skin Economy Research Assistant. Goodbye!")
            break
        
        # Skip empty queries
        if not query.strip():
            continue
        
        try:
            print(f"\nResearching your query using {search_type.lower()} search...")
            raw_response = agent_executor.invoke({"input": query})
            
            try:
                structured_response = parser.parse(raw_response.get("output", ""))
                print(f"\n{structured_response.summary}")
            except Exception as e:
                print(f"\nError parsing structured response: {str(e)}")
                print(f"\nRaw response: {raw_response.get('output', 'No output')}")
        
        except KeyboardInterrupt:
            print("\nOperation canceled by user.")
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main() 