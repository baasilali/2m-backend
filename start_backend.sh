#!/bin/bash

# Set color variables
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================================${NC}"
echo -e "${GREEN}Starting CS2 Skin Economy API${NC}"
echo -e "${BLUE}==================================================${NC}"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo -e "${GREEN}Virtual environment found. Activating...${NC}"
    source venv/bin/activate
else
    echo -e "${YELLOW}No virtual environment found. Creating one...${NC}"
    python -m venv venv
    source venv/bin/activate
    echo -e "${GREEN}Installing dependencies...${NC}"
    pip install -r requirements.txt
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}No .env file found. Creating a template...${NC}"
    echo "OPENAI_API_KEY=your_api_key_here" > .env
    echo -e "${RED}Please edit the .env file to add your OpenAI API key.${NC}"
fi

# Start the API server
echo -e "${GREEN}Starting API server...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
uvicorn main:app --reload --port 8000 