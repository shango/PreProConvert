# Use ASWF (Academy Software Foundation) image with VFX libraries pre-built
# Includes PyAlembic, imath, OpenEXR, USD, etc.
FROM aswf/ci-vfxall:2023-clang15.2

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Default port (Railway overrides via $PORT)
ENV PORT=8000
EXPOSE 8000

# Run with uvicorn - use shell form to expand $PORT
CMD uvicorn web.app:app --host 0.0.0.0 --port $PORT
