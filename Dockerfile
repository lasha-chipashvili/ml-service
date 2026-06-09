FROM python:3.11-slim

WORKDIR /app

# 1. Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy your actual project directories into the container
COPY API/ ./API/
COPY ML/ ./ML/

# 3. Expose the port FastAPI runs on
EXPOSE 8000

# 4. Run Uvicorn pointing to your app inside the API folder
CMD ["uvicorn", "API.app:app", "--host", "0.0.0.0", "--port", "8000"]