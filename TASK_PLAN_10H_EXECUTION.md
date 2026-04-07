# KE HOACH THUC THI 10 TIENG - AI INTERIOR DESIGN

Muc tieu: dua du an ve trang thai demo an toan + bao cao on dinh + co bang chung ky thuat.

Tong thoi gian: 10 gio tap trung.

## Nguyen tac thuc hien

- Lam theo thu tu uu tien P0 -> P1 -> P2.
- Moi block xong phai commit nho.
- Neu block nao vuong > 20 phut thi chuyen block tiep theo, ghi lai blocker.
- Luon giu duoc 1 branch build chay duoc de demo.

## MOC 0 - CHUAN BI (00:00 -> 00:30)

### Task 0.1 - Tao nhanh checklist tong
- [done] Tao branch moi: fix/report-readiness
- [done] Chot 3 luong demo:
  - [done] Upload + Segmentation (ổn ở mức cơ bản cho SAM nhưng với SAM 3 thì việc nhập liệu ví dụ ghế nhưng cái mục ấn để bắt đầu phân đoạn lại bị tô xám không ấn được)
  - [done] Inpainting remove object (dùng lama nên đôi lúc nó còn mờ ở chỗ xóa, nhưng nhìn chung là ổn)
  -[done] Generation design (style/placement) (ổn nhưng thời gian tùy vào style mà người dùng chọn, có style nhanh có style chậm)
- [ổn] Chot script test nhanh cuoi moi block

### Task 0.2 - Chup hien trang truoc khi sua
- [done] Chay: git status
- [done] Chay: backend lenh health check (neu backend da run)
- [done] Chup anh/ghi chu loi hien tai de doi chieu sau fix

Tieu chi hoan thanh MOC 0:
- Co branch rieng
- Co checklist demo 3 luong
- Co moc baseline

---

## MOC 1 - P0: FIX LOI CHAN DEMO (00:30 -> 02:30)

### Task 1.1 - Sua endpoint legacy predict cho dung contract
Files:
- backend/app/main.py

Viec can lam:
- [done] Sua ham predict_legacy nhan UploadFile = File(...)
- [done] Dam bao endpoint /predict dung contract (tranh 422 do sai kieu file)
- [done] Dam bao backward compatibility voi frontend cu

Check nhanh:
- [done] POST /predict voi 1 image => 200 hoac loi nghiep vu ro rang (khong crash contract)

### Task 1.2 - Dong bo cau hinh inpainting method
Files:
- backend/app/config.py
- backend/.env.example
- backend/app/core/inpainting_service.py

Viec can lam:
- [done] Chot 1 bien chinh: REMOVE_OBJECT_METHOD
- [done] Ho tro alias INPAINTING_METHOD tam thoi de khong vo moi truong cu
- [done] Cap nhat comment huong dan trong .env.example

Check nhanh:
- [done] Dat REMOVE_OBJECT_METHOD=lama chay dung
- [done] Dat REMOVE_OBJECT_METHOD=replicate chay dung

### Task 1.3 - Bo sung dependency thieu trong requirements
Files:
- backend/requirements.txt

Viec can lam:
- [done] Bo comment torch/torchvision trong requirements
- [done] Dam bao setup clone moi khong mat buoc thu cong bat ngo

Check nhanh:
- [done] pip install -r requirements.txt khong vo loi dependency chinh

### Task 1.4 - Don luong frontend de tranh bam nham API cu
Files:
- ../interior_ai/frontend/lib/main.dart
- ../interior_ai/frontend/lib/config.dart

Viec can lam:
- [done] Bo nut TEST OLD API khoi man hinh chinh
- [done] Giam rui ro demo bam nham

Check nhanh:
- [done] Tu man hinh chinh chi con luong khuyen nghi

Tieu chi hoan thanh MOC 1:
- 3 loi critical da xu ly
- Demo luong chinh khong crash vi contract/config
- [done] MOC 1 hoan tat (code + verify)

Commit de xuat:
- fix(p0): legacy predict + config unification + deps + frontend legacy cleanup

---

## MOC 2 - P1: ON DINH VA TEST TOI THIEU (02:30 -> 05:30)

### Task 2.1 - Sua test Flutter template sai app
Files:
- ../interior_ai/frontend/test/widget_test.dart
- ../interior_ai/frontend/lib/main.dart

Viec can lam:
- [done] Thay test mau Counter/MyApp bang test dung app hien tai
- [done] It nhat co 1 smoke test build app thanh cong

Check nhanh:
- [done] flutter test pass toi thieu 1 test

### Task 2.2 - Bo sung smoke test backend
Files de tao moi:
- backend/tests/test_health.py
- backend/tests/test_segmentation_contract.py
- backend/tests/test_inpainting_job_contract.py

Viec can lam:
- [done] Test /api/v1/health tra schema co status
- [done] Test contract segmentation request/response co field can thiet
- [done] Test inpainting async tra job_id va status hop le

Check nhanh:
- [done] pytest backend/tests -q

### Task 2.3 - Don debug print/log khong can thiet
Files:
- ../interior_ai/frontend/lib/screens/inpainting_screen.dart
- backend/app/services/replicate_client.py

Viec can lam:
- [done] Thay print bang logger hoac xu ly im lang co chu dich
- [done] Dam bao log de doc khi demo

### Task 2.4 - Xac nhan phu thuoc Redis trong runbook
Files:
- docker-compose.yml
- backend/README.md

Viec can lam:
- [done] Ghi ro remove-object-async can Redis
- [done] Co lenh run chuan local: docker compose up -d redis + run backend

Tieu chi hoan thanh MOC 2:
- [done] MOC 2 hoan tat (code + test + docs)

Commit de xuat:
- test(p1): add smoke tests + docs(redis) + logging cleanup

---

## MOC 3 - P2: NANG CHAT BAO CAO (05:30 -> 08:00)

### Task 3.1 - Dong bo tai lieu tien do voi code that
Files:
- PROJECT_PROGRESS_REPORT.md
- README.md
- backend/README.md

Viec can lam:
- [done] Cap nhat ty le tien do that
- [done] Bo/doi cac dong Week 3 = 0% neu da co generation implementation
- [done] Ghi ro phan nao MVP, phan nao production-ready

### Task 3.2 - Tao bang ket qua thu nghiem ngan
File de tao moi:
- docs/EXPERIMENT_RESULTS.md

Viec can lam:
- [todo-verify] Chay 5-10 anh mau
- [done] Ghi: latency, success rate, chi phi cloud uoc tinh, ghi chu quality (baseline)
- [done] Tach ket qua theo method (lama, replicate, local) (baseline)

### Task 3.3 - Viet file trade-off
File de tao moi:
- docs/TRADEOFFS_AND_LIMITATIONS.md

Viec can lam:
- [done] Neu ro vi sao dung fallback chain
- [done] Neu ro gioi han in-memory jobs o generation
- [done] Neu ro han che GPU 4GB va tac dong

Tieu chi hoan thanh MOC 3:
- Tai lieu khop code
- Co so lieu de tra loi hoi dong
- Co phan han che/de xuat phat trien ro rang
- [partial] MOC 3 hoan tat phan tai lieu; con benchmark 5-10 anh de chot so lieu cuoi

Commit de xuat:
- docs(p2): progress sync + experiment results + tradeoffs

---

## MOC 4 - CHOT DEMO VA BAO VE (08:00 -> 10:00)

### Task 4.1 - Dry-run demo 2 vong
- [ ] Vong 1: full happy path
- [ ] Vong 2: mo phong loi token/mang, dung fallback
- [ ] Ghi thoi gian tung buoc

### Task 4.2 - Tao demo script
File de tao moi:
- DEMO_SCRIPT.md

Noi dung bat buoc:
- [ ] Thu tu thao tac 5-7 phut
- [ ] Input mau
- [ ] Ket qua mong doi
- [ ] Ke hoach backup neu fail mang/API

### Task 4.3 - Chot slide noi dung ky thuat
- [ ] Dong gop chinh
- [ ] Kien truc + pipeline
- [ ] So lieu ket qua
- [ ] Han che + huong phat trien

Tieu chi hoan thanh MOC 4:
- Demo script ro rang
- Co phuong an backup
- Co storyline bao ve mach lac

Commit de xuat:
- chore: final demo script and defense readiness

---

## DANH SACH KIEM THU CUOI CUNG (GO/NO-GO)

### Backend
- [ ] /api/v1/health OK
- [ ] /api/v1/segmentation/upload OK
- [ ] /api/v1/segmentation/segment-points OK
- [ ] /api/v1/inpainting/remove-object-async OK
- [ ] /api/v1/inpainting/job-status/{job_id} OK
- [ ] /api/v1/generation/generate-design OK
- [ ] /api/v1/generation/job-status/{job_id} OK

### Frontend
- [ ] Chay duoc tu man hinh chinh den segmentation
- [ ] Sau inpainting xem duoc ket qua
- [ ] Chuyen qua generation va nhan ket qua
- [ ] Khong con nut/luong de bam nham gay fail

### Tai lieu
- [ ] README khop trang thai hien tai
- [ ] PROJECT_PROGRESS_REPORT khop implementation
- [ ] Co EXPERIMENT_RESULTS
- [ ] Co DEMO_SCRIPT

### Test
- [ ] flutter test pass toi thieu smoke
- [ ] pytest pass toi thieu smoke

Neu con duoi 70% checklist o gio thu 8:
- Cat pham vi: uu tien luong Segmentation + Inpainting + 1 style generation
- Dong bang tinh nang moi, tap trung on dinh demo.

---

## MAU LOG THEO DOI TIEN DO (DIEN TAY)

- [time] Bat dau task:
- [time] Ket qua:
- [time] Blocker:
- [time] Cach xu ly:
- [time] Commit hash:

