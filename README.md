# AI Interior Design - Computer Vision Project

He thong ho tro cai tao noi that bang Computer Vision + Generative AI.

## Tong Quan

Du an gom 2 phan:
- Backend FastAPI (Python) trong thu muc `backend/` thuoc workspace `interior_project`
- Frontend Flutter trong workspace khac: `/mnt/d/interior_ai/frontend`

Muc tieu chinh:
1. Tach doi tuong noi that can xoa (Segmentation)
2. Xoa doi tuong va tao phong trong (Inpainting)
3. Tao thiet ke moi theo style (Generation)

## Trang Thai Hien Tai (04/2026)

| Module | Trang thai MVP | Production-ready | Ghi chu |
|---|---|---|---|
| Segmentation (SAM local + SAM3 cloud) | Co | Chua | SAM3 cloud hien tai uu tien text prompt |
| Inpainting (LaMa/Replicate/Local fallback) | Co | Chua | Async can Redis, chat luong phu thuoc mask |
| Generation (ControlNet + placement) | Co | Chua | Job generation dang luu in-memory |
| AR | Chua | Chua | De o scope tuong lai |

## Kien Truc Tong The

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

## Quick Start

### 1) Backend

```bash
cd /home/tran_thien/interior_project/backend
pip install -r requirements.txt
cp .env.example .env
```

Neu dung endpoint async inpainting (`/api/v1/inpainting/remove-object-async`) thi can Redis:

```bash
cd /home/tran_thien/interior_project
docker compose up -d redis
```

Chay backend:

```bash
cd /home/tran_thien/interior_project/backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2) Frontend

```bash
cd /mnt/d/interior_ai/frontend
flutter pub get
flutter run
```

## Luong Demo De Xuat

1. Upload anh va segmentation doi tuong
2. Remove object (inpainting)
3. Generate style moi hoac placement noi that

## Tai Lieu Quan Trong

- [Tien do cap nhat](PROJECT_PROGRESS_REPORT.md)
- [Backend README](backend/README.md)
- [Ket qua thu nghiem](docs/EXPERIMENT_RESULTS.md)
- [Trade-offs va gioi han](docs/TRADEOFFS_AND_LIMITATIONS.md)
- [Networking setup](NETWORKING_ARCHITECTURE.md)

## Luu Y Scope

- Ban hien tai da dat muc tieu bao cao do an o muc MVP.
- Chua phai phien ban production-ready vi con cac gioi han ve persistence, test coverage sau, va hardening.

## License

Academic Project - For Educational Purposes
