# AI Interior Design - Computer Vision Project

Hệ thống hỗ trợ thiết kế và tái cấu trúc nội thất sử dụng Generative AI và AR.

## 📋 Mục tiêu Dự án

Xây dựng ứng dụng cho phép:
1. **Xóa bỏ** vật thể nội thất cũ bằng AI (SAM + Stable Diffusion Inpainting)
2. **Tạo sinh** thiết kế mới với ControlNet
3. **Trực quan hóa** trong AR với tỉ lệ thực tế

## 🏗️ Kiến trúc Hệ thống

```
┌─────────────────┐
│  Flutter App    │  (Mobile - Android/iOS)
│  - Camera       │
│  - AR View      │
└────────┬────────┘
         │ HTTP/REST
         ▼
┌─────────────────┐
│  FastAPI Server │  (Backend - Python)
│  - SAM          │  (WSL Ubuntu + GTX 1650)
│  - Diffusion    │
│  - ControlNet   │
└─────────────────┘
```

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

### Backend Setup

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Move model weights
mv sam_vit_b_01ec64.pth weights/

# Run server
python -m app.main
```

Server sẽ chạy tại: `http://localhost:8000`

### Frontend Setup

```bash
cd frontend

# Install dependencies
flutter pub get

# Run app
flutter run
```

### Port Forwarding (Windows → WSL)

```powershell
# Run as Administrator
.\setup_port_forward.ps1
```

## 🎯 Roadmap 4 Tuần

### Tuần 1: SAM Segmentation ✅ (In Progress)
- [x] Restructure codebase
- [ ] Interactive segmentation với click points
- [ ] Generate và save masks

### Tuần 2: Inpainting Pipeline
- [ ] Tích hợp Stable Diffusion Inpainting
- [ ] Optimize prompts cho "empty room"
- [ ] Flutter: Hiển thị kết quả

### Tuần 3: ControlNet Generation
- [ ] MLSD/Canny edge detection
- [ ] ControlNet integration
- [ ] Style selection UI

### Tuần 4: AR + Finalization
- [ ] ARCore basic placement
- [ ] Báo cáo và documentation
- [ ] Video demo

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI
- **AI Models**: 
  - SAM (Segment Anything Model)
  - Stable Diffusion Inpainting
  - ControlNet
  - MLSD (optional)
- **Hardware**: GTX 1650 4GB (CUDA)

### Frontend
- **Framework**: Flutter
- **Plugins**: 
  - image_picker
  - http
  - arcore_flutter_plugin (planned)

## 📊 SWOT Analysis

**Strengths**: Giải quyết pain point thực sự (renovation vs add-on), tech stack hiện đại

**Weaknesses**: VRAM 4GB hạn chế, thời gian 4 tuần gấp

**Opportunities**: Cloud APIs (Replicate, HuggingFace), pretrained models

**Threats**: Diffusion models nặng, AR trên Flutter còn bug

## 📖 Documentation

- [Backend README](backend/README.md)
- [Project Objectives](.kiro/steering/project-objectives.md)
- [Project Structure](.kiro/steering/project-structure.md)

## 🤝 Contributing

Dự án môn học - Computer Vision, Năm 4

## 📝 License

Academic Project - For Educational Purposes
