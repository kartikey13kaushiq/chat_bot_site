FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8000
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src src
RUN pip install --no-cache-dir . && useradd --system app
USER app
EXPOSE 8000
# Configure the model with CHAT_PROVIDER / CHAT_MODEL / *_API_KEY (see README); defaults to offline demo mode.
CMD ["python", "-m", "chatsite"]
