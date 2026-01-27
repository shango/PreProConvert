# Use ASWF (Academy Software Foundation) image with VFX libraries pre-built
FROM aswf/ci-vfxall:2023-clang15.2 AS builder

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .

# Fix requirements - remove wrong alembic package (PyAlembic is pre-installed in ASWF image)
RUN grep -v "^alembic>=" requirements.txt > requirements-fixed.txt && \
    pip install --no-cache-dir -r requirements-fixed.txt

# Production stage
FROM aswf/ci-vfxall:2023-clang15.2

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy application code
COPY . .

# Remove wrong alembic from requirements for any future installs
RUN grep -v "^alembic>=" requirements.txt > requirements-fixed.txt && \
    mv requirements-fixed.txt requirements.txt

# Expose port
EXPOSE 8000

# Run with uvicorn
CMD ["python", "-m", "uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
