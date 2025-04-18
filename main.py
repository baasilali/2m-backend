from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import AgentExecutor, create_openai_functions_agent
from tools import search_tool, wiki_tool, save_tool, cs_skins_tool
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os

# Try to import the search engine - this might fail if dependencies are missing
try:
    from search_utils import get_skin_search_engine
    has_search_engine = True
except ImportError:
    try:
        from search_utils_fallback import get_skin_search_engine
        has_search_engine = True
    except ImportError:
        has_search_engine = False
        print("WARNING: Could not import either search engine - search functionality will be limited")

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow cross-origin requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the search engine at app startup
@app.on_event("startup")
async def startup_event():
    # Initialize the search engine to preload embeddings
    if has_search_engine:
        try:
            print("Initializing skin search engine...")
            search_engine = get_skin_search_engine()
            print(f"Search engine initialized with {len(search_engine.item_names)} items")
        except Exception as e:
            print(f"Error initializing search engine: {str(e)}")
    else:
        print("Search engine not available - search functionality will be limited")

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

llm = ChatOpenAI(model="gpt-4o-mini")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a research assistant that specializes in the Counter Strike skin economy and marketplace.
    You have access to both web search tools and a local database containing third-party marketplace
    data for CS:GO skins from SkinPort, DMarket, and CSFloat.
    
    For skin prices and marketplace information, always prefer to use the local cs_skins tool 
    before searching online.
    
    Answer the user query and use necessary tools. 
    Wrap the output in this format and provide no other text\n{format_instructions}
    """),
    ("human", "{input}"),
    ("assistant", "{agent_scratchpad}")
]).partial(format_instructions=parser.get_format_instructions())

tools = [cs_skins_tool, search_tool, wiki_tool, save_tool]
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
    Endpoint to handle queries from the frontend.
    Runs the agent executor with the user's query and returns the response.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        print(f"Received query: {request.query}")
        raw_response = agent_executor.invoke({"input": request.query})
        
        try:
            structured_response = parser.parse(raw_response.get("output", ""))
            # Return the structured response in a format expected by the frontend
            return QueryResponse(
                answer=structured_response.summary,
                sources=structured_response.sources
            )
        except Exception as e:
            print(f"Error parsing structured response: {str(e)}")
            # Fallback: Return the raw output
            return QueryResponse(
                answer=raw_response.get("output", "Sorry, I couldn't process that request."),
                sources=[]
            )
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "CS2 Skin Economy API is running!"}

def main():
    print("CS:GO Skin Economy Research Assistant")
    print("Type 'exit', 'quit', or 'q' to end the session")
    
    while True:
        query = input("\nWhat can I help you with? ")
        
        # Check for exit command
        if query.lower() in ['exit', 'quit', 'q']:
            print("Thank you for using the CS:GO Skin Economy Research Assistant. Goodbye!")
            break
        
        # Skip empty queries
        if not query.strip():
            continue
        
        try:
            print("\nResearching your query...")
            raw_response = agent_executor.invoke({"input": query})
            
            try:
                structured_response = parser.parse(raw_response.get("output", ""))
                # Only show the summary in terminal mode
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