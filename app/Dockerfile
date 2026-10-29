FROM python:3.12-slim

# Run as a regular user, not root
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    FASTEMBED_CACHE=/home/user/model_cache
WORKDIR /home/user/app

# Install packages first, so code changes don't trigger a full reinstall
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the ONNX embedding model during the build, so the first visitor doesn't wait for it
RUN python -c "import os; from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir=os.environ['FASTEMBED_CACHE'])"

COPY --chown=user . .

# Render tells the app which port to use through the PORT variable
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}
