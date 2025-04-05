from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import AgentExecutor, create_openai_functions_agent
from tools import search_tool, wiki_tool, save_tool, cs_skins_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    

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
                print_response(structured_response)
            except Exception as e:
                print(f"\nError parsing structured response: {str(e)}")
                print(f"\nRaw response: {raw_response.get('output', 'No output')}")
        
        except KeyboardInterrupt:
            print("\nOperation canceled by user.")
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main()