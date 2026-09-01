#!/bin/sh
set -e

MODEL="${MODEL:-qwen3:8b}"

# Start the Ollama server in the background
ollama serve &
SERVER_PID=$!

# Wait for the server to become ready
until ollama list >/dev/null 2>&1; do
  sleep 1
done

# Pull the model if it isn't already present
if ! ollama list | grep -q "^${MODEL}\s"; then
  echo "Model ${MODEL} not found - pulling..."
  ollama pull "${MODEL}"
else
  echo "Model ${MODEL} already present."
fi

# Keep the server in the foreground
wait "${SERVER_PID}"