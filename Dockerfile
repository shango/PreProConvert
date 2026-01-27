# Use ASWF (Academy Software Foundation) image with VFX libraries pre-built
# Includes PyAlembic, imath, OpenEXR, USD, etc.
FROM aswf/ci-vfxall:2023-clang15.2

WORKDIR /app

# Copy and install Python dependencies (with cache mount for faster rebuilds)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Default port (Railway/Fly.io override via $PORT)
ENV PORT=8000
EXPOSE 8000

# Run with uvicorn - use /bin/bash to ensure variable expansion works
CMD ["/bin/bash", "-c", "uvicorn web.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
