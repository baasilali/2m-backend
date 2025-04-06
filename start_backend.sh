#!/bin/bash

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
  echo "Activating virtual environment..."
  source .venv/bin/activate
elif [ -d "venv" ]; then
  echo "Activating virtual environment..."
  source venv/bin/activate
fi

# Install requirements
echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir

# Remove the flag file since we're always installing requirements now
if [ -f ".requirements_installed" ]; then
  rm .requirements_installed
fi

# Start the FastAPI server
echo "Starting FastAPI server..."
uvicorn main:app --reload --host 0.0.0.0 --port 8000 