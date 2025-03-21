# Python AI Research Agent

An intelligent research assistant built with LangChain that helps gather and structure information on any topic using various tools including search and Wikipedia.

## Features

- Interactive research assistant
- Structured output using Pydantic models
- Multiple tool integration (Search, Wikipedia, Save)
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

Enter your research query when prompted, and the agent will gather and structure information for you.

## Project Structure

- `main.py`: Main application entry point
- `tools.py`: Custom tools for the agent
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (not included in repo)

## License

MIT License