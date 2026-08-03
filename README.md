# AI Interior Design - Computer Vision Project

Hệ thống hỗ trợ thiết kế và tái cấu trúc nội thất sử dụng Generative AI và AR.

## 📋 Mục tiêu Dự án

Xây dựng ứng dụng cho phép:
1. **Xóa bỏ** vật thể nội thất cũ bằng AI (SAM + Stable Diffusion Inpainting)
2. **Tạo sinh** thiết kế mới với ControlNet
3. **Trực quan hóa** trong AR với tỉ lệ thực tế

## 🏗️ Kiến trúc Hệ thống

```text
Flutter App (Android/iOS)
        |
        | HTTP/REST
        v
FastAPI Backend (WSL/Linux)
  - SAM segmentation (local/cloud)
  - Inpainting service (lama/replicate/local)
  - ControlNet generation
  - Redis job persistence (inpainting/generation/placement async)
```

## 📊 Trạng Thái Hiện Tại (04/2026)

| Module | Trang thai MVP | Production-ready | Ghi chu |
|---|---|---|---|
| Segmentation (SAM local + SAM3 cloud) | ✅ Có | ⏳ Chưa | SAM3 cloud hiện tại ưu tiên text prompt |
| Inpainting (LaMa/Replicate/Local fallback) | ✅ Có | ⏳ Chưa | Async dùng Redis, chất lượng phụ thuộc mask |
| Generation (ControlNet + placement) | ✅ Có | ⏳ Chưa | Job generation/placement dùng Redis |
| AR | ⏳ Chưa | ⏳ Chưa | Để ở scope tương lai |

## 📁 Cấu trúc Dự án

```
interior_ai/backend/
├── backend/              # Python FastAPI Backend
│   ├── app/             # Application code
│   ├── tests/           # Backend smoke/contract tests
│   ├── weights/         # Model checkpoints
│   ├── data/            # Data storage
│   └── Dockerfile
├── docs/                # Documentation
└── docker-compose.yml   # Redis + backend service
```

## 🚀 Quick Start

### 1) Backend Setup

```bash
cd /home/tran_thien/workspace/interior_ai/backend/backend
pip install -r requirements.txt
cp .env.example .env
```

Redis native trong WSL là mặc định cho local development:

```bash
sudo systemctl enable --now redis-server
redis-cli -h 127.0.0.1 -p 6379 ping
```

Redis Docker là lựa chọn phụ và được map sang host port `6380`:

```bash
cd /home/tran_thien/workspace/interior_ai/backend
docker compose up -d redis
REDIS_URL=redis://127.0.0.1:6380/0 \
  python -m uvicorn --app-dir backend app.main:app --host 0.0.0.0 --port 8000
```

Chạy server từ thư mục backend canonical:
```bash
cd /home/tran_thien/workspace/interior_ai/backend/backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Swagger UI: `http://localhost:8000/docs`

### 2) Frontend Setup

```bash
cd /home/tran_thien/workspace/interior_ai/frontend
flutter pub get
flutter run -d web-server --web-hostname=0.0.0.0 --web-port=8080 \
  --dart-define=API_BASE_URL=http://localhost:8000
```

### 3) Port Forwarding (Windows → WSL)

```powershell
# Run as Administrator
.\setup_port_forward.ps1
```

## 🎯 Roadmap 4 Tuần

### Tuần 1: SAM Segmentation ✅
- [x] Restructure codebase
- [x] Interactive segmentation với click points
- [x] Generate và save masks

### Tuần 2: Inpainting Pipeline ✅
- [x] Tích hợp Stable Diffusion Inpainting
- [x] Optimize prompts cho "empty room"
- [x] Flutter: Hiển thị kết quả

### Tuần 3: ControlNet Generation ✅
- [x] MLSD/Canny edge detection
- [x] ControlNet integration
- [x] Style selection UI

### Tuần 4: AR + Finalization
- [ ] ARCore basic placement
- [ ] Báo cáo và documentation
- [ ] Video demo

## 🛠️ Tech Stack

- **Backend**: FastAPI, SAM, Stable Diffusion, ControlNet, Redis
- **Frontend**: Flutter (image_picker, http, arcore)
- **Hardware**: GTX 1650 4GB (CUDA)

## 📖 Documentation

- [Backend README](backend/README.md)
- [Tiến độ cập nhật](PROJECT_PROGRESS_REPORT.md)
- [Kết quả thử nghiệm](docs/EXPERIMENT_RESULTS.md)
- [Trade-offs và giới hạn](docs/TRADEOFFS_AND_LIMITATIONS.md)
- [Networking setup](NETWORKING_ARCHITECTURE.md)

## 🤝 Contributing

Dự án môn học - Computer Vision, Năm 4

## 📝 License

Academic Project - For Educational Purposes
