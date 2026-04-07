---
inclusion: always
---

# Cấu trúc Thư mục Chuyên nghiệp - AI Interior Design

## Cấu trúc Đề xuất

```
interior_project/
├── backend/                          # Python FastAPI Backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI app entry point
│   │   ├── config.py                 # Configuration (API keys, paths, etc.)
│   │   ├── dependencies.py           # Dependency injection
│   │   │
│   │   ├── api/                      # API Routes
│   │   │   ├── __init__.py
│   │   │   ├── v1/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── endpoints/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── segmentation.py    # SAM endpoints
│   │   │   │   │   ├── inpainting.py      # Inpainting endpoints
│   │   │   │   │   ├── generation.py      # ControlNet endpoints
│   │   │   │   │   └── health.py          # Health check
│   │   │   │   └── router.py
│   │   │
│   │   ├── core/                     # Core business logic
│   │   │   ├── __init__.py
│   │   │   ├── sam_segmentation.py   # SAM wrapper
│   │   │   ├── diffusion_inpainting.py  # Stable Diffusion wrapper
│   │   │   ├── controlnet_generation.py # ControlNet wrapper
│   │   │   ├── mlsd_detection.py     # MLSD wrapper (optional)
│   │   │   └── pipeline.py           # End-to-end pipeline orchestration
│   │   │
│   │   ├── models/                   # Pydantic models (request/response)
│   │   │   ├── __init__.py
│   │   │   ├── segmentation.py
│   │   │   ├── inpainting.py
│   │   │   └── generation.py
│   │   │
│   │   ├── services/                 # External services
│   │   │   ├── __init__.py
│   │   │   ├── replicate_client.py   # Replicate API client
│   │   │   ├── huggingface_client.py # HuggingFace API client
│   │   │   └── storage.py            # File storage service
│   │   │
│   │   └── utils/                    # Utilities
│   │       ├── __init__.py
│   │       ├── image_processing.py   # Image utils (resize, convert, etc.)
│   │       ├── mask_utils.py         # Mask manipulation
│   │       └── logger.py             # Logging setup
│   │
│   ├── weights/                      # Model checkpoints
│   │   ├── sam_vit_b_01ec64.pth
│   │   └── .gitkeep
│   │
│   ├── data/                         # Data storage
│   │   ├── inputs/                   # User uploaded images
│   │   ├── outputs/                  # Generated results
│   │   ├── masks/                    # Generated masks
│   │   └── temp/                     # Temporary files
│   │
│   ├── tests/                        # Unit tests
│   │   ├── __init__.py
│   │   ├── test_sam.py
│   │   ├── test_inpainting.py
│   │   └── test_api.py
│   │
│   ├── notebooks/                    # Jupyter notebooks for experiments
│   │   ├── test_sam.ipynb
│   │   ├── test_inpainting.ipynb
│   │   └── benchmark.ipynb
│   │
│   ├── requirements.txt              # Python dependencies
│   ├── requirements-dev.txt          # Dev dependencies (pytest, jupyter, etc.)
│   ├── .env.example                  # Environment variables template
│   ├── .gitignore
│   └── README.md
│
├── frontend/                         # Flutter Mobile App
│   ├── lib/
│   │   ├── main.dart
│   │   │
│   │   ├── config/                   # App configuration
│   │   │   ├── api_config.dart       # API endpoints
│   │   │   └── theme.dart            # App theme
│   │   │
│   │   ├── models/                   # Data models
│   │   │   ├── room_image.dart
│   │   │   ├── segmentation_result.dart
│   │   │   └── design_style.dart
│   │   │
│   │   ├── services/                 # API & Business logic
│   │   │   ├── api_service.dart      # HTTP client wrapper
│   │   │   ├── image_service.dart    # Image picker & processing
│   │   │   └── ar_service.dart       # AR functionality
│   │   │
│   │   ├── screens/                  # UI Screens
│   │   │   ├── home_screen.dart
│   │   │   ├── camera_screen.dart
│   │   │   ├── segmentation_screen.dart
│   │   │   ├── result_screen.dart
│   │   │   └── ar_screen.dart
│   │   │
│   │   ├── widgets/                  # Reusable widgets
│   │   │   ├── image_preview.dart
│   │   │   ├── loading_indicator.dart
│   │   │   └── style_selector.dart
│   │   │
│   │   └── utils/                    # Utilities
│   │       ├── constants.dart
│   │       └── helpers.dart
│   │
│   ├── assets/                       # Static assets
│   │   ├── images/
│   │   ├── icons/
│   │   └── 3d_models/                # 3D models for AR
│   │
│   ├── test/                         # Widget tests
│   ├── pubspec.yaml
│   └── README.md
│
├── docs/                             # Documentation
│   ├── api/                          # API documentation
│   │   └── openapi.yaml
│   ├── architecture.md               # System architecture
│   ├── setup.md                      # Setup guide
│   └── deployment.md                 # Deployment guide
│
├── scripts/                          # Utility scripts
│   ├── setup_environment.sh          # Environment setup
│   ├── download_models.py            # Download model weights
│   └── benchmark.py                  # Performance benchmarking
│
├── .kiro/                            # Kiro configuration
│   └── steering/
│       ├── project-objectives.md
│       └── project-structure.md
│
├── setup_port_forward.ps1            # Port forwarding script
├── docker-compose.yml                # Docker setup (optional)
├── .gitignore
└── README.md                         # Project overview
```

## Nguyên tắc Tổ chức

### Backend (Python/FastAPI)

1. **Separation of Concerns**:
   - `api/`: Chỉ handle HTTP requests/responses
   - `core/`: Business logic thuần túy, không biết gì về HTTP
   - `services/`: Tương tác với external services (APIs, databases)
   - `models/`: Data validation và serialization

2. **Dependency Injection**:
   - Models được load 1 lần trong `dependencies.py`
   - Inject vào endpoints qua FastAPI dependencies

3. **Versioning**:
   - API v1 trong `api/v1/`
   - Dễ dàng thêm v2 sau này mà không break v1

### Frontend (Flutter)

1. **Clean Architecture**:
   - `models/`: Pure data classes
   - `services/`: Business logic & API calls
   - `screens/`: Full-page widgets
   - `widgets/`: Reusable components

2. **Single Responsibility**:
   - Mỗi screen chỉ lo 1 chức năng
   - Services tách biệt (API, Image, AR)

### Data Management

1. **Organized Storage**:
   - `data/inputs/`: Raw user uploads
   - `data/outputs/`: Final results
   - `data/masks/`: Intermediate masks
   - `data/temp/`: Auto-cleanup temporary files

2. **Model Weights**:
   - Tách riêng folder `weights/`
   - Không commit vào git (dùng .gitignore)
   - Script download tự động

## Migration Plan (Từ cấu trúc hiện tại)

### Phase 1: Backend Restructure
1. Tạo folder structure mới
2. Di chuyển `main.py` → `app/main.py`
3. Tách logic SAM ra `core/sam_segmentation.py`
4. Di chuyển checkpoint → `weights/`
5. Tạo `config.py` cho configuration

### Phase 2: API Modularization
1. Tạo endpoints riêng biệt
2. Implement Pydantic models
3. Add proper error handling
4. Add logging

### Phase 3: Frontend Restructure
1. Tạo folder structure
2. Tách `main.dart` thành screens
3. Tạo API service layer
4. Implement proper state management

## Benefits

✅ **Scalability**: Dễ thêm features mới (MLSD, ControlNet) mà không ảnh hưởng code cũ
✅ **Maintainability**: Mỗi file có trách nhiệm rõ ràng
✅ **Testability**: Dễ viết unit tests cho từng module
✅ **Collaboration**: Nhiều người có thể làm việc song song
✅ **Professional**: Cấu trúc chuẩn industry, dễ present cho giáo viên

## Notes

- Không cần implement tất cả ngay, làm dần theo roadmap
- Ưu tiên core features trước (SAM, Inpainting)
- Có thể bỏ qua một số phần (tests, docs) nếu thiếu thời gian
- Cấu trúc này scale được cho production sau khi tốt nghiệp
