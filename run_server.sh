#!/bin/bash

# Mentra + Suno HackMIT Backend Server Runner
# This script helps you set up and run the server

echo "🚀 Starting Mentra + Suno HackMIT Backend Server"
echo "================================================"

# Check if SUNO_API_KEY is set
if [ -z "$SUNO_API_KEY" ]; then
    echo "⚠️  SUNO_API_KEY not set!"
    echo "Please set your Suno API key:"
    echo "  export SUNO_API_KEY='your_api_key_here'"
    echo ""
    echo "Or create a .env file with:"
    echo "  echo 'SUNO_API_KEY=your_api_key_here' > .env"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "✅ SUNO_API_KEY is set"
echo "🌐 Starting server on http://localhost:8000"
echo "📖 API docs available at http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python3 main.py
