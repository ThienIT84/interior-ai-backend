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
  - Redis job persistence (inpainting async)
```

## 📊 Trạng Thái Hiện Tại (04/2026)

| Module | Trang thai MVP | Production-ready | Ghi chu |
|---|---|---|---|
| Segmentation (SAM local + SAM3 cloud) | ✅ Có | ⏳ Chưa | SAM3 cloud hiện tại ưu tiên text prompt |
| Inpainting (LaMa/Replicate/Local fallback) | ✅ Có | ⏳ Chưa | Async cần Redis, chất lượng phụ thuộc mask |
| Generation (ControlNet + placement) | ✅ Có | ⏳ Chưa | Job generation đang lưu in-memory |
| AR | ⏳ Chưa | ⏳ Chưa | Để ở scope tương lai |

## 📁 Cấu trúc Dự án

```
interior_project/
├── backend/              # Python FastAPI Backend
│   ├── app/             # Application code
│   ├── weights/         # Model checkpoints
│   ├── data/            # Data storage
│   └── notebooks/       # Experiments
│
├── frontend/            # Flutter Mobile App
│   └── lib/            # Dart code
│
├── docs/               # Documentation
├── scripts/            # Utility scripts
└── .kiro/             # Kiro AI configuration
```

## 🚀 Quick Start

### 1) Backend Setup

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
```

**Lưu ý**: Nếu dùng endpoint async inpainting (`/api/v1/inpainting/remove-object-async`) thì cần chạy Redis:

```bash
# Tại thư mục gốc project
docker compose up -d redis
```

Chạy server:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Swagger UI: `http://localhost:8000/docs`

### 2) Frontend Setup

```bash
cd frontend
flutter pub get
flutter run
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
