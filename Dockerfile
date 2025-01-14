# Use Python 3.10 slim image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy necessary files
COPY chatbot.json ./chatbot.json
COPY requirements.txt ./requirements.txt
COPY utils/ ./utils/
COPY main.py ./main.py

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create faiss_indices directory
RUN mkdir faiss_indices

# Expose the port FastAPI will run on
EXPOSE 8000

# Start the FastAPI app
CMD ["python", "main.py"]
