# AI Interior Design - Backend

Backend API cho hệ thống AI Interior Design sử dụng FastAPI, SAM, và Stable Diffusion.

## 🎯 Features

- ✅ **SAM Segmentation**: Interactive point-based object segmentation
- ✅ **Stable Diffusion Inpainting**: Remove objects and generate empty rooms
- ⏳ **ControlNet Generation**: Generate new interior designs (Week 3)
- ⏳ **AR Support**: 3D visualization endpoints (Week 4)

## 📊 Performance

| Component | Latency | VRAM | Cost |
|-----------|---------|------|------|
| SAM Segmentation | 0.2-0.5s | 1.5-2GB | $0 |
| SD Inpainting | 13-15 min | 3.5-4GB | $0 |

**Hardware**: GTX 1650 4GB VRAM

## 🚀 Quick Start

### 1. Activate Environment
```bash
~/miniconda3/envs/interior_ai/bin/python
```

### 2. Start Server
```bash
cd ~/interior_project/backend
~/miniconda3/envs/interior_ai/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Access API
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 4. Redis for Async Inpainting (Required)

`POST /api/v1/inpainting/remove-object-async` stores and tracks jobs in Redis.

Start Redis from project root:

```bash
cd ~/interior_project
docker compose up -d redis
```

Verify Redis is reachable:

```bash
redis-cli -h 127.0.0.1 -p 6379 ping
# Expected: PONG
```

If Redis is not available, use synchronous endpoint `POST /api/v1/inpainting/remove-object`.

## 📁 Project Structure

```
backend/
├── app/                          # Main application
│   ├── api/v1/endpoints/         # API routes
│   │   ├── segmentation.py       # ✅ SAM endpoints
│   │   ├── inpainting.py         # ✅ Inpainting endpoints
│   │   └── health.py             # ✅ Health check
│   ├── core/                     # Business logic
│   │   ├── sam_segmentation.py   # ✅ SAM wrapper
│   │   └── diffusion_inpainting.py # ✅ SD wrapper
│   ├── config.py                 # Configuration
│   └── main.py                   # FastAPI app
├── data/                         # Data storage
│   ├── inputs/                   # Uploaded images
│   ├── masks/                    # Generated masks
│   └── outputs/                  # Inpainting results
├── weights/                      # Model checkpoints
│   └── sam_vit_b_01ec64.pth     # SAM model
└── Test scripts & Documentation
```

## 🔧 Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (GTX 1n650 4GB or better)
- Conda environment: `interior_ai`

### Installation

```bash
# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. Download SAM weights (if not exists)
# Place sam_vit_b_01ec64.pth in weights/

# 3. Configure environment (optional)
cp .env.example .env
```

### First Run

```bash
# Start server
~/miniconda3/envs/interior_ai/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Test in another terminal
~/miniconda3/envs/interior_ai/bin/python test_inpainting_api.py
```

## 📡 API Endpoints

### Segmentation
- `POST /api/v1/segmentation/upload` - Upload image, get image_id
- `POST /api/v1/segmentation/segment-points` - Segment with points
- `GET /api/v1/segmentation/image/{image_id}` - Get uploaded image
- `GET /api/v1/segmentation/mask-image/{mask_id}` - Get mask PNG

### Inpainting
- `POST /api/v1/inpainting/remove-object-async` - Submit inpainting job
- `GET /api/v1/inpainting/job-status/{job_id}` - Check job status
- `GET /api/v1/inpainting/result/{result_id}` - Get result image

### Health
- `GET /api/v1/health` - System health check

## 🧪 Testing

```bash
# Test SAM segmentation
~/miniconda3/envs/interior_ai/bin/python test_sam_dataset.py

# Test inpainting integration
~/miniconda3/envs/interior_ai/bin/python test_inpainting_api.py

# Benchmark performance
~/miniconda3/envs/interior_ai/bin/python benchmark_sam.py

# Optimize parameters
~/miniconda3/envs/interior_ai/bin/python optimize_inpainting_strength.py
```

See [TESTING.md](TESTING.md) for detailed test guide.

## 📚 Documentation

- **[DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)** - Development workflow
- **[TESTING.md](TESTING.md)** - Testing guide
- **[TEST_INPAINTING.md](TEST_INPAINTING.md)** - Inpainting test instructions
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Current project status
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture
- **[COST_ANALYSIS.md](COST_ANALYSIS.md)** - Cost comparison
- **[INPAINTING_ALTERNATIVES.md](INPAINTING_ALTERNATIVES.md)** - Alternative methods

## 🛠️ Development

### Adding New Features

1. **Core logic**: Add to `app/core/`
2. **API models**: Add to `app/models/`
3. **Endpoints**: Add to `app/api/v1/endpoints/`
4. **Register**: Update `app/api/v1/router.py`

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings
- Handle errors gracefully

## ⚙️ Configuration

### Optimized Parameters (GTX 1650 4GB)

**SAM:**
```python
model = "vit_b"
checkpoint = "weights/sam_vit_b_01ec64.pth"
```

**Stable Diffusion Inpainting:**
```python
model = "runwayml/stable-diffusion-inpainting"
dtype = torch.float32  # NOT fp16
steps = 50
guidance_scale = 11.0
strength = 0.99
```

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check Python version
~/miniconda3/envs/interior_ai/bin/python --version

# Check dependencies
~/miniconda3/envs/interior_ai/bin/pip list | grep -E "torch|diffusers|fastapi"
```

### Out of memory
```bash
# Check VRAM usage
nvidia-smi

# Clear CUDA cache
~/miniconda3/envs/interior_ai/bin/python -c "import torch; torch.cuda.empty_cache()"
```

### Black output from inpainting
✅ **FIXED**: Use float32 instead of fp16 (implemented in code)

See [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) for more troubleshooting.

## 📈 Roadmap

- [x] **Week 1**: SAM segmentation + Flutter UI
- [x] **Week 2 (Day 8-9)**: Inpainting integration
- [ ] **Week 2 (Day 10-14)**: Testing & optimization
- [ ] **Week 3**: ControlNet generation
- [ ] **Week 4**: AR + finalization

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for detailed progress.

## 🤝 Contributing

1. Follow project structure
2. Write tests for new features
3. Update documentation
4. Test on real images

## 📝 Notes

- Models load once at startup (singleton pattern)
- CORS enabled for Flutter app
- Auto-detect CUDA/CPU
- Results cached in `data/outputs/`
- Async processing for long-running tasks

## 📞 Support

For issues:
1. Check documentation files
2. Review test scripts
3. Check API docs at `/docs`
4. Review code comments
