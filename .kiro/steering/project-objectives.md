---
inclusion: always
---

# Mục tiêu Đề tài: AI Interior Design - Computer Vision Project

## Thông tin Dự án
- **Môn học**: Thị giác máy tính (Computer Vision)
- **Cấp độ**: Sinh viên năm 4
- **Thời gian còn lại**: 4 tuần
- **Hardware**: GTX 1650 4GB VRAM (giới hạn)

## Mục tiêu Tổng quát
Xây dựng hệ thống hỗ trợ thiết kế và tái cấu trúc nội thất toàn diện:
- Xóa bỏ hiện trạng bừa bộn của không gian thực bằng pipeline AI xóa vật thể và tái tạo nền
- Trực quan hóa ý tưởng thiết kế mới qua workflow generate lại phòng theo style

## Pipeline Kỹ thuật Cập nhật (03/2026)

### Flow chính (đã chốt)
1. **Segmentation**: SAM local + SAM3 (prompt text) để tạo mask vật thể
2. **Object Removal**: LaMa inpainting để xóa vật thể và giữ nền liền mạch
3. **Redesign**: ControlNet + Canny edge để dựng lại không gian theo style
4. **Refine (optional)**: Inpainting pass 2 chỉ cho vùng lỗi cục bộ

### Nguyên tắc triển khai
- Tách 2 mask:
   - `mask_remove`: vùng xóa vật thể (dilation 8-20 px)
   - `mask_restyle`: vùng được phép redesign (có thể rộng hơn)
- Có quality gate trước/sau LaMa (coverage mask, artifact biên)
- Canny dùng chế độ adaptive theo edge density mục tiêu 5-15%
- Retry có điều kiện (đổi seed/tinh chỉnh tham số), tối đa 2 lần

## 4 Mục tiêu Kỹ thuật Cốt lõi (Updated Status)

### 🎯 Mục tiêu 1: Scene Understanding (Hiểu cấu trúc không gian)
**Công nghệ**: MLSD (Mobile Line Segment Detection)
- Phát hiện đường biên, góc tường, trần nhà
- Xác định ma trận phối cảnh (Perspective Transformation)
- Đảm bảo vật thể ảo đặt đúng mặt phẳng sàn

**Trạng thái**: ❌ Chưa bắt đầu
**Ưu tiên**: Trung bình (có thể đơn giản hóa)

### 🎯 Mục tiêu 2: Semantic Inpainting (Tái cấu trúc hình ảnh)
**Công nghệ**: SAM/SAM3 + LaMa + Inpainting fallback
- SAM/SAM3: Phân đoạn vùng vật thể (point + text prompt)
- LaMa: Xóa vật thể ưu tiên chất lượng nền
- Inpainting fallback: Sửa lỗi cục bộ khi LaMa còn artifact

**Trạng thái**: ✅ Đã hoạt động end-to-end (cần tối ưu quality gate)
**Ưu tiên**: ⭐⭐⭐ CAO NHẤT (core feature)

### 🎯 Mục tiêu 3: Generative Design (Tạo sinh ý tưởng)
**Công nghệ**: ControlNet + Stable Diffusion
- Sinh phương án phối màu và sắp xếp nội thất mới
- Dựa trên layout sẵn có của căn phòng thông qua Canny edge

**Trạng thái**: ✅ Đã có API + async job + style presets
**Ưu tiên**: ⭐⭐ Cao (cần cho demo)

### 🎯 Mục tiêu 4: AR Visualization (Trực quan hóa tương tác)
**Công nghệ**: ARCore + SLAM
- Duy trì vị trí vật thể ảo khi di chuyển camera
- Sai số đo đạc mặt phẳng < 5cm (mục tiêu lý tưởng)

**Trạng thái**: ❌ Chưa bắt đầu
**Ưu tiên**: ⭐ Thấp (có thể làm proof-of-concept đơn giản)

## Deliverables (Sản phẩm cần nộp)

1. **Backend AI Server**: 
   - API nhận ảnh thô → phân đoạn → remove object → redesign
   - Thời gian xử lý mục tiêu:
     - remove-only: 5-15 giây (cloud path)
     - full redesign: 15-45 giây
   - Trạng thái: ✅ Pipeline chính đã có, cần harden production

2. **Mobile App (Flutter)**:
   - Chụp ảnh, segmentation tương tác, theo dõi job inpainting/generation
   - Trạng thái: ✅ Đã tích hợp các bước AI chính; AR chưa triển khai

3. **Báo cáo & Tài liệu**:
   - Quy trình pipeline mới (SAM/SAM3 -> LaMa -> ControlNet + Canny)
   - Tham số tối ưu và chiến lược fallback/retry
   - Đánh giá hiệu năng và chi phí theo backend
   - Trạng thái: 🟡 Đang tổng hợp

## SWOT Analysis

### Strengths (Điểm mạnh)
- ✅ Đề tài có tính thực tiễn cao, giải quyết pain point thực sự
- ✅ USP rõ ràng: Renovation (xóa cũ-xây mới) vs Add-on (chỉ thêm đồ)
- ✅ Kết hợp nhiều kỹ thuật CV tiên tiến phù hợp năm 4
- ✅ Đã có infrastructure cơ bản (FastAPI + Flutter + GPU setup)
- ✅ Kiến trúc client-server đúng hướng cho AI workload

### Weaknesses (Điểm yếu)
- ⚠️ GTX 1650 4GB VRAM quá nhỏ cho Stable Diffusion (cần 6-8GB)
- ⚠️ Thời gian còn lại chỉ 4 tuần, phạm vi quá rộng
- ⚠️ Chưa có kinh nghiệm với Diffusion models và AR
- ⚠️ Backend trên WSL có thể gặp vấn đề performance
- ⚠️ Chất lượng đầu ra nhạy với chất lượng mask và prompt

### Opportunities (Cơ hội)
- 💡 Có thể dùng API cloud (Replicate, Hugging Face) để giải quyết vấn đề VRAM
- 💡 Flutter có plugin ARCore sẵn, không cần code native
- 💡 SAM và MLSD có pretrained models tốt, không cần train
- 💡 Có thể đơn giản hóa AR thành proof-of-concept để tiết kiệm thời gian
- 💡 Diffusion models có thể chạy với reduced precision (fp16) để fit 4GB

### Threats (Rủi ro)
- 🚨 Stable Diffusion có thể không chạy được trên GTX 1650 4GB
- 🚨 Inpainting/generation quality phụ thuộc mạnh vào mask quality và edge quality
- 🚨 AR trên Flutter còn nhiều bug, có thể cần fallback sang native
- 🚨 MLSD có thể fail với góc chụp quá nghiêng hoặc ánh sáng kém
- 🚨 Thời gian 4 tuần không đủ để hoàn thiện cả 4 mục tiêu

## Roadmap 4 Tuần (Điều chỉnh theo tiến độ thực tế)

### Tuần 1 (đã hoàn thành): Nền tảng API + Segmentation
- [x] FastAPI architecture + endpoint v1
- [x] SAM local integration
- [x] SAM3 Replicate integration (text-prompt path)
- [x] Upload/image-mask storage flow

### Tuần 2 (đã hoàn thành): Object Removal Pipeline
- [x] Hybrid inpainting service (local/cloud)
- [x] LaMa object removal integration
- [x] Async job endpoint cho inpainting
- [x] Flutter flow segmentation -> inpainting

### Tuần 3 (đang tối ưu): Generative Design + Quality Control
- [x] Canny preview endpoint
- [x] ControlNet async generation endpoint
- [x] Style presets + generation UI
- [ ] Quality gate cho mask/remove/generation
- [ ] Inpainting pass 2 cho vùng lỗi cục bộ
- [ ] Auto retry theo score

### Tuần 4: AR PoC + Báo cáo + Demo cuối
- [ ] ARCore plugin cơ bản: đặt 1 vật thể 3D
- [ ] Hoàn thiện benchmark (thời gian, chất lượng, chi phí)
- [ ] Chốt báo cáo kỹ thuật + video demo
- [ ] Chuẩn bị presentation và kịch bản demo

## Chiến lược Giảm Rủi ro

### Về VRAM (4GB limitation):
**Giải pháp**: Ưu tiên cloud cho bước nặng, local cho fallback
- Replicate API: ~$0.01/image, latency 5-10s
- LaMa trên Replicate: nhanh, chi phí thấp cho object removal
- Fallback local: chỉ dùng khi cloud không khả dụng

### Về chất lượng mask và edge:
**Giải pháp**: Quality gate bắt buộc trước khi generate
- Mask coverage gate: reject nếu quá nhỏ/quá lớn
- Morphology + feather trước khi remove
- Adaptive Canny với edge density mục tiêu 5-15%
- Chỉ chạy pass 2 tại vùng lỗi, tránh degrade toàn ảnh

### Về AR complexity:
**Giải pháp**: Đơn giản hóa thành proof-of-concept
- Chỉ cần plane detection + place 1 object
- Không cần persistent tracking (SLAM)
- Acceptable error: 10-15cm thay vì 5cm

### Về thời gian:
**Giải pháp**: Focus vào core features
- Mục tiêu 2 (Semantic Inpainting): MUST HAVE - đã xong bản nền, đang tối ưu
- Mục tiêu 3 (Generative Design): MUST HAVE cho demo - đã xong bản nền, đang tối ưu
- Mục tiêu 1 (MLSD): NICE TO HAVE
- Mục tiêu 4 (AR): DEMO ONLY

## Success Criteria (Tiêu chí thành công)

### Minimum Viable Product (MVP):
- ✅ User chụp ảnh -> chọn object -> remove object (LaMa)
- ✅ Backend generate 1-3 phong cách thiết kế từ ảnh đã làm sạch
- ✅ Flutter app theo dõi async jobs và hiển thị kết quả
- 🟡 AR demo 1 vật thể 3D (nếu kịp)

### Stretch Goals (Nếu còn thời gian):
- 🎯 Multiple object selection trong 1 ảnh
- 🎯 Style transfer với nhiều options
- 🎯 AR với 5-10 mẫu đồ nội thất
- 🎯 Save/load design history
- 🎯 Auto quality scoring + auto retry pipeline

## Notes cho Development

- **Luôn test trên ảnh thực tế**, không chỉ test trên ảnh mẫu
- **Mask quality quan trọng hơn prompt** trong bước remove
- **Measure latency + cost** ở mỗi bước (segmentation/remove/generate)
- **Fallback plan**: Cloud fail -> local; quality thấp -> pass 2 cục bộ
- **Document everything**: Mỗi quyết định kỹ thuật cần có lý do rõ ràng cho báo cáo

## USP (Unique Selling Point) - Nhấn mạnh trong thuyết minh

> "Các ứng dụng hiện nay chỉ giải quyết bài toán Add-on (thêm đồ vào). 
> Dự án này giải quyết bài toán Renovation (xóa cũ - xây mới), 
> đây là pain point lớn nhất của người dùng khi muốn cải tạo nhà cửa bừa bộn."
