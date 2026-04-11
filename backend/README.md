# AI Interior Design - Backend

Backend API cho he thong AI Interior Design (FastAPI + SAM + Inpainting + ControlNet).

## Current Delivery Status

| Capability | MVP | Production-ready | Notes |
|---|---|---|---|
| Segmentation API | Yes | No | Local SAM + SAM3 cloud mode |
| Inpainting API (sync/async) | Yes | No | Async inpainting requires Redis |
| Generation API | Yes | No | Some generation jobs are in-memory |
| Automated smoke tests | Yes | No | Baseline coverage only |

## Core Features

- Segmentation:
  - `POST /api/v1/segmentation/upload`
  - `POST /api/v1/segmentation/segment-points`
  - `POST /api/v1/segmentation/segment-box`
- Inpainting:
  - `POST /api/v1/inpainting/remove-object`
  - `POST /api/v1/inpainting/remove-object-async`
  - `GET /api/v1/inpainting/job-status/{job_id}`
  - `GET /api/v1/inpainting/result/{result_id}`
- Generation:
  - `POST /api/v1/generation/preview-edges`
  - `POST /api/v1/generation/generate-design`
  - `GET /api/v1/generation/job-status/{job_id}`
  - `POST /api/v1/generation/place-furniture`

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional but recommended for local path)
- Redis (required for async inpainting jobs)

## Setup

```bash
cd /home/tran_thien/interior_project/backend
pip install -r requirements.txt
cp .env.example .env
```

Place SAM checkpoint in `weights/`:

```text
backend/weights/sam_vit_b_01ec64.pth
```

## Run

Start Redis from project root:

```bash
cd /home/tran_thien/interior_project
docker compose up -d redis
```

For host-run backend, use this in `backend/.env`:

```text
REDIS_URL=redis://127.0.0.1:6380/0
```

Quick verify:

```bash
redis-cli -h 127.0.0.1 -p 6380 ping
redis-cli -h 127.0.0.1 -p 6380 --scan --pattern 'interior_job:*'
```

Run backend:

```bash
cd /home/tran_thien/interior_project/backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API docs:
- Swagger: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing

Run smoke tests:

```bash
cd /home/tran_thien/interior_project/backend
python -m pytest tests -q
```

Current smoke test scope:
- Health contract
- Segmentation request/schema contract
- Async inpainting job submission contract

## Configuration Notes

- Preferred inpainting method config: `REMOVE_OBJECT_METHOD`
- Legacy alias still supported: `INPAINTING_METHOD`
- Segmentation mode: `SEGMENTATION_BACKEND=local|sam3_replicate`

## Known Limitations

1. Generation job persistence is not fully Redis-backed yet.
2. Inpainting quality depends strongly on mask quality.
3. Local SD path is slow on 4GB VRAM GPUs.

## References

- Root progress report: `../PROJECT_PROGRESS_REPORT.md`
- Experiment log: `../docs/EXPERIMENT_RESULTS.md`
- Trade-offs: `../docs/TRADEOFFS_AND_LIMITATIONS.md`
