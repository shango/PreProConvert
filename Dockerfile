# Runtime Dockerfile - uses pre-built PyAlembic from lib/
FROM debian:bookworm-slim

# Install Python and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    libboost-python1.74.0 \
    libimath-3-1-29 \
    libopenexr-3-1-30 \
    libhdf5-103-1 \
    python3-imath \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pre-built PyAlembic libraries (built locally with Dockerfile.build)
COPY lib/ /usr/local/lib/

# Set library paths
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PYTHONPATH=/usr/local/lib/python3.11/site-packages

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy application code
COPY . .

# Default port
ENV PORT=8000
EXPOSE 8000

CMD ["/bin/bash", "-c", "python3 -m uvicorn web.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
