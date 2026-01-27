# PreProConvert

Convert Alembic, USD, and Maya files to multiple formats via a web interface.

## Supported Formats

**Input:**
- Alembic (.abc)
- USD (.usd, .usda, .usdc)
- Maya ASCII (.ma)

**Output:**
- After Effects (.jsx, .obj)
- USD (.usdc)
- Maya MA (.ma)
- FBX (.fbx)

## Running Locally

Requires native VFX libraries (PyAlembic, imath). Use Docker for easiest setup.

### With Docker

```bash
docker build -t preproconvert .
docker run -p 8000:8000 preproconvert
```

Then open http://localhost:8000

### Without Docker

Requires PyAlembic and imath installed on your system.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn web.app:app --host 0.0.0.0 --port 8000
```

## Deployment

### Railway (recommended)

1. Push to GitHub
2. Connect repo at [railway.app](https://railway.app)
3. Railway auto-detects Dockerfile and deploys

### Other Docker-compatible hosts

Any platform supporting Docker deployments will work (Fly.io, AWS ECS, Google Cloud Run, etc.)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload a scene file |
| GET | `/api/jobs/{id}` | Get job status |
| POST | `/api/jobs/{id}/convert` | Start conversion |
| GET | `/api/jobs/{id}/progress` | SSE progress stream |
| GET | `/api/jobs/{id}/download` | Download results (ZIP) |
| DELETE | `/api/jobs/{id}` | Delete job |
| GET | `/api/formats` | List supported formats |
