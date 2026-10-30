FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir numpy scipy scikit-learn fastapi uvicorn pydantic pyjwt
COPY src ./src
COPY data ./data
ENV PYTHONPATH=/app/src
# ERAG_JWT_SECRET must be provided at runtime (>=16 chars); the server refuses to start without it
EXPOSE 8000
CMD ["python", "-m", "erag.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
