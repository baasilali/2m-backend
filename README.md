# CS:GO Skin Economy Research Assistant

An intelligent research assistant built with LangChain that helps gather and structure information on the Counter Strike skin economy and marketplace, using both a local database of third-party marketplace data and online sources.

## Features

- Interactive research assistant focused on CS:GO skin prices and marketplace data
- Local database for skin prices from popular third-party marketplaces:

  - SkinPort
  - DMarket
  - CSFloat
  - Buff163
  - HaloSkins

- Web search capabilities for supplementary information
- Structured output using Pydantic models
- Multiple tool integration (CS:GO Skins Database, Search, Wikipedia, Save)
- Clean and maintainable code structure

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd PythonAIAgentFromScratch
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
```

## Usage

Run the main script:
```bash
python main.py
```

Enter your CS:GO skin economy-related query when prompted. The agent will first check the local marketplace database for information, and if needed, supplement with online research.

### Sample Queries

- "How much is AWP Dragon Lore on SkinPort?"
- "Compare prices for Karambit Doppler across marketplaces"
- "Show me all available knife skins"
- "What are the cheapest gloves on the market?"
- "List all AK-47 skins"
- "What are the most expensive skins available?"
- "Show me all StatTrak items"
- "Compare marketplace average prices"

## Project Structure

- `main.py`: Main application entry point
- `tools.py`: Custom tools for the agent
- `data/prices_output.json`: Third-party marketplace data for CS:GO skins
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (not included in repo)

## Extending the Database

The marketplace data is stored in `data/prices_output.json`. This contains price information from SkinPort, DMarket, and CSFloat for various CS:GO skins. You can update this file with the latest prices or add more marketplaces as needed.

## Technical Implementation

### LangChain Architecture
- **Agent Framework**: Uses LangChain's AgentExecutor and OpenAI Functions Agent for tool selection and execution
- **LLM Integration**: Powered by OpenAI's GPT-4o mini model, optimized for cost-effective performance
- **Tool Integration**: Custom and built-in LangChain tools with a priority routing system to prefer local database queries

### Structured Data Processing
- **Pydantic Models**: Response data is structured using Pydantic for type safety and consistency
- **Output Parsing**: PydanticOutputParser ensures consistent response format with topic, summary, sources, and tools used
- **Error Handling**: Multi-level exception handling for robust operation when parsing fails

### Marketplace Data Processing
- **Intelligent Query Parsing**: Natural language understanding to extract marketplace names, item types, and query intents
- **Price Normalization**: Automatic conversion between different marketplace price formats (USD conversion from other currencies)
- **Context-Aware Responses**: Tailors responses based on query specificity, providing detailed data for precise queries and summaries for broader ones

### Performance Optimizations
- **Lazy Loading**: Database is loaded only when needed to minimize startup time
- **Result Limiting**: Pagination-like approach to prevent overwhelming responses
- **Cached Responses**: No redundant computations when analyzing marketplace data

## License

MIT License