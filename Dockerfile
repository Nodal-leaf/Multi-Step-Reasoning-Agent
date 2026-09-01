FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY agent.py .
COPY reasoning_agent/ ./reasoning_agent/

ENTRYPOINT ["python", "agent.py"]
