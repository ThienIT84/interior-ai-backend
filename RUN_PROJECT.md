# 🚀 Hướng Dẫn Chạy Dự Án - AI Interior Design

## 📋 Tổng quan kiến trúc

```
┌──────────────────────┐         HTTP/REST        ┌─────────────────────────┐
│  📱 Flutter App      │ ◄──── localhost:8000 ───► │  🐍 FastAPI Backend     │
│  (Windows)           │      (ADB reverse)        │  (WSL Ubuntu)           │
│  D:\interior_ai\     │                           │  ~/interior_project/    │
│  frontend\           │                           │  backend/               │
└──────────────────────┘                           └─────────────────────────┘
                                                          │
                                                          ▼
                                                   ┌─────────────┐
                                                   │ GTX 1650 4GB│
                                                   │ (CUDA)      │
                                                   └─────────────┘
```

---

## 1. Prerequisites (Cài đặt 1 lần)

### Trên WSL (Ubuntu)
- **Python 3.10+** via Miniconda/Anaconda
- **CUDA toolkit** (đã cài sẵn cho GTX 1650)
- **Conda environment**: `interior_ai`

### Trên Windows
- **Flutter SDK** (3.10+)
- **Android Studio** + Android SDK
- **ADB** (đi kèm Android Studio)
- **VS Code** (khuyến nghị)

---

## 2. Chạy Backend (WSL Ubuntu)

### Bước 1: Mở terminal WSL
```bash
# Mở Ubuntu terminal hoặc dùng VS Code WSL terminal
cd ~/interior_project/backend
```

### Bước 2: Kích hoạt môi trường conda
```bash
conda activate interior_ai
```

### Bước 3: Cài dependencies (lần đầu hoặc khi có thay đổi)
```bash
pip install -r requirements.txt
```

### Bước 4: Cấu hình .env
```bash
# Copy file mẫu (lần đầu)
cp .env.example .env

# Chỉnh sửa .env với API keys thực tế
nano .env
```

Nội dung `.env` cần thiết:
```env
REPLICATE_API_TOKEN=your_token_here
HUGGINGFACE_API_TOKEN=your_token_here
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
SAM_CHECKPOINT_PATH=weights/sam_vit_b_01ec64.pth
```

### Bước 5: Chạy backend

**Cách 1 — Script ổn định (khuyến nghị cho demo):**
```bash
bash run_backend_stable.sh
```

**Cách 2 — Chạy trực tiếp:**
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Cách 3 — Development mode (auto-reload):**
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Bước 6: Kiểm tra backend đã chạy
```bash
# Trong WSL, mở terminal mới
curl http://localhost:8000/docs
# Hoặc mở trình duyệt Windows: http://localhost:8000/docs
```

> ⚠️ **Lưu ý**: Lần đầu khởi động sẽ mất 2-3 phút để load model SAM + Stable Diffusion.

---

## 3. Kết nối Điện thoại → Backend

### Cách 1: ADB Reverse (khuyến nghị — ổn định nhất)

Yêu cầu: Điện thoại kết nối USB + bật USB Debugging.

**Trên Windows (PowerShell hoặc CMD):**
```bash
# Kiểm tra điện thoại đã kết nối
adb devices

# Tạo reverse port forwarding
adb reverse tcp:8000 tcp:8000
```

✅ Sau khi setup, Flutter app dùng `http://localhost:8000` — không cần biết IP!

### Cách 2: WiFi (khi không có cáp USB)

**Bước 1 — Lấy IP của WSL:**
```bash
# Trong WSL
hostname -I
# Kết quả ví dụ: 172.22.105.141
```

**Bước 2 — Port forwarding Windows → WSL (chạy PowerShell Admin):**
```powershell
# Dùng script có sẵn (cập nhật IP trong file trước)
.\setup_port_forward.ps1

# Hoặc chạy thủ công:
netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=<WSL_IP>
```

**Bước 3 — Cập nhật IP trong Flutter:**
Mở `frontend/lib/config.dart`, uncomment dòng WiFi IP:
```dart
// return "http://localhost:8000";         // ADB reverse
return "http://192.168.1.XX:8000";         // WiFi - thay XX bằng IP Windows
```

---

## 4. Chạy Frontend (Windows)

### Bước 1: Mở terminal Windows
```bash
cd D:\interior_ai\frontend
```

### Bước 2: Cài dependencies (lần đầu hoặc khi có thay đổi)
```bash
flutter pub get
```

### Bước 3: Kết nối điện thoại
```bash
# Kiểm tra thiết bị
flutter devices
```

### Bước 4: Chạy app
```bash
# Debug mode
flutter run

# Release mode (mượt hơn, khuyến nghị cho demo)
flutter run --release
```

---

## 5. Quy trình chạy nhanh (sau khi setup xong)

```
Terminal 1 (WSL):                       Terminal 2 (Windows):
─────────────────                       ──────────────────────
cd ~/interior_project/backend           cd D:\interior_ai\frontend
conda activate interior_ai             adb reverse tcp:8000 tcp:8000
bash run_backend_stable.sh              flutter run --release
                                        
↓ Chờ "✅ Models loaded"                ↓ App chạy trên điện thoại
↓ Chờ "✅ SD model pre-loaded"          ↓ Sẵn sàng sử dụng!
```

---

## 6. Cấu trúc thư mục sau dọn dẹp

```
interior_project/                     # Root (WSL)
├── .kiro/                           # Kiro specs & steering docs
├── backend/                         # Python FastAPI Backend
│   ├── app/                        # 🔥 Core application code
│   │   ├── main.py                 #   Entry point
│   │   ├── config.py               #   Configuration
│   │   ├── dependencies.py         #   Dependency injection
│   │   ├── api/v1/                 #   API endpoints
│   │   ├── core/                   #   Business logic (SAM, Inpainting)
│   │   ├── models/                 #   Pydantic models
│   │   ├── services/               #   External services
│   │   └── utils/                  #   Utilities
│   ├── weights/                    # Model checkpoints (~375MB)
│   ├── data/                       # Runtime data (inputs/outputs/masks)
│   ├── notebooks/                  # Jupyter experiments
│   ├── .env                        # API keys (KHÔNG commit)
│   ├── requirements.txt            # Python dependencies
│   └── run_backend_stable.sh       # Stable run script
├── README.md
├── setup_adb.sh                    # ADB reverse setup script
└── setup_port_forward.ps1          # Port forwarding script

D:\interior_ai\frontend\             # Frontend (Windows)
├── lib/
│   ├── main.dart                   # App entry point
│   ├── config.dart                 # API URL config
│   ├── screens/                    # UI screens
│   ├── services/                   # API service layer
│   └── models/                     # Data models
├── android/                        # Android build config
├── pubspec.yaml                    # Flutter dependencies
└── ...
```

---

## 7. Troubleshooting

| Vấn đề | Giải pháp |
|---------|-----------|
| Backend không khởi động | Kiểm tra `conda activate interior_ai` đã chạy chưa |
| SAM model not found | Kiểm tra file `weights/sam_vit_b_01ec64.pth` tồn tại |
| Điện thoại không kết nối được | Chạy lại `adb reverse tcp:8000 tcp:8000` |
| Timeout khi gọi API | Lần đầu load model mất 2-3 phút, chờ log "✅ Models loaded" |
| Flutter build lỗi | Chạy `flutter clean` rồi `flutter pub get` lại |
| Port 8000 bị chiếm | `lsof -i :8000` (WSL) hoặc `netstat -ano | findstr 8000` (Windows) |
| CUDA out of memory | Restart backend để free VRAM |

---

## 8. API Endpoints chính

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/docs` | Swagger UI documentation |
| `POST` | `/api/v1/segmentation/segment` | Upload ảnh + SAM segmentation |
| `POST` | `/api/v1/segmentation/segment-points` | Segment theo click points |
| `POST` | `/api/v1/inpainting/remove-object` | Xóa vật thể (sync) |
| `POST` | `/api/v1/inpainting/remove-object-async` | Xóa vật thể (async) |
| `GET` | `/api/v1/inpainting/job-status/{job_id}` | Kiểm tra trạng thái job |
