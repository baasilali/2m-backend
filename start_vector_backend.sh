#!/bin/bash
echo "Starting CS2 Skin Economy API with Vector Search..."

# Check if we're in the right directory
if [ ! -f "main_vector.py" ]; then
    echo "Error: main_vector.py not found. Please run from the 2m-backend directory."
    exit 1
fi

# Start the server
echo "🚀 Starting vector-enhanced API server..."
python main_vector.py
