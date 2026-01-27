#!/bin/bash
# Build PyAlembic in Docker and extract the libraries
# Run this once, then commit the lib/ directory

set -e

echo "Building PyAlembic in Docker..."

docker build -f Dockerfile.build -t pyalembic-builder .

echo "Extracting built libraries..."

# Create lib directory
mkdir -p lib

# Copy libraries from the builder container
docker create --name pyalembic-extract pyalembic-builder
docker cp pyalembic-extract:/usr/local/lib/. lib/
docker rm pyalembic-extract

echo "Done! Libraries extracted to lib/"
echo "Now commit the lib/ directory and update Dockerfile to use it"
