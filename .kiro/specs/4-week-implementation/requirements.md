# Requirements: 4-Week AI Interior Design Implementation

## 1. Executive Summary

Triển khai hệ thống AI Interior Design trong 4 tuần với phạm vi thực tế, tập trung vào core features có tính khả thi cao trên hardware GTX 1650 4GB VRAM.

**Chiến lược chính:** Hybrid approach - SAM local + Diffusion cloud API + AR simplified

## 2. SWOT Analysis (Chi tiết)

### 2.1 Strengths (Điểm mạnh)

#### S1: Infrastructure đã sẵn sàng
- **Mô tả**: Backend structure chuyên nghiệp với FastAPI, dependency injection, API versioning
- **Impact**: Tiết kiệm 3-4 ngày setup time
- **Evidence**: 
  - ✅ `app/` structure với separation of concerns
  - ✅ SAM model đã load được
  - ✅ Port forwarding Windows ↔ WSL đã setup
- **Leverage**: Có thể focus 100% vào AI logic thay vì infrastructure

#### S2: Tech stack phù hợp với academic project
- **Mô tả**: Kết hợp nhiều kỹ thuật CV tiên tiến (SAM, Diffusion, ControlNet, AR)
- **Impact**: Thể hiện được kiến thức đa dạng, dễ ghi điểm với giáo viên
- **Evidence**: 
  - SAM: State-of-the-art segmentation (Meta AI 2023)
  - Stable Diffusion: Leading generative model
  - ControlNet: Conditional generation (ICCV 2023)
- **Leverage**: Mỗi component có thể viết thành 1 section riêng trong báo cáo

#### S3: USP rõ ràng
- **Mô tả**: Renovation (xóa cũ - xây mới) vs Add-on (chỉ thêm đồ)
- **Impact**: Differentiation mạnh so với các đề tài khác
- **Evidence**: Không có app nào trên thị trường giải quyết bài toán này end-to-end
- **Leverage**: Dễ pitch, dễ demo, dễ gây ấn tượng

#### S4: Flutter cross-platform
- **Mô tả**: Code 1 lần chạy được cả Android/iOS
- **Impact**: Tiết kiệm thời gian development
- **Evidence**: 
  - Xiaomi 13 Pro (Android) là target chính
  - Có thể demo trên iOS nếu cần
- **Leverage**: Có thể scale sau khi tốt nghiệp

#### S5: Pretrained models sẵn có
- **Mô tả**: SAM, SD, ControlNet đều có pretrained weights chất lượng cao
- **Impact**: Không cần training, chỉ cần inference
- **Evidence**: 
  - SAM vit_b: 375MB, accuracy cao
  - SD v1.5: Proven quality
- **Leverage**: Focus vào integration thay vì model training

### 2.2 Weaknesses (Điểm yếu)

#### W1: VRAM 4GB quá nhỏ cho local Diffusion
- **Mô tả**: GTX 1650 4GB không đủ để chạy SD v1.5 (cần 6-8GB)
- **Impact**: HIGH - Không thể chạy full pipeline local
- **Evidence**: 
  - SD v1.5 base: ~4GB VRAM
  - SD v1.5 + ControlNet: ~6GB VRAM
  - SAM vit_b: ~2GB VRAM
  - Total: 8GB+ (vượt quá 4GB)
- **Mitigation**: 
  - ✅ Dùng cloud API (Replicate, HuggingFace)
  - ✅ Chỉ chạy SAM local
  - ⚠️ Fallback: OpenCV inpainting nếu API fail

#### W2: Thời gian 4 tuần rất gấp
- **Mô tả**: Phạm vi ban đầu (6 tuần) bị compress xuống 4 tuần
- **Impact**: HIGH - Risk không hoàn thành đầy đủ features
- **Evidence**: 
  - Tuần 1: SAM + Infrastructure (feasible)
  - Tuần 2: Inpainting pipeline (challenging)
  - Tuần 3: ControlNet (challenging)
  - Tuần 4: AR + Report (very tight)
- **Mitigation**:
  - ✅ Cut AR scope xuống proof-of-concept only
  - ✅ Dùng API thay vì implement local
  - ✅ Parallel work: Backend + Frontend

#### W3: Chưa có kinh nghiệm với Diffusion models
- **Mô tả**: Chưa từng làm việc với SD, ControlNet, prompt engineering
- **Impact**: MEDIUM - Learning curve cao
- **Evidence**: 
  - Prompt engineering khó (cần nhiều thử nghiệm)
  - ControlNet conditioning phức tạp
  - Debugging diffusion models khó
- **Mitigation**:
  - ✅ Dùng API (abstraction layer)
  - ✅ Copy prompts từ community (Civitai, Reddit)
  - ✅ Focus vào integration, không đi sâu vào model internals

#### W4: AR trên Flutter còn nhiều limitations
- **Mô tả**: ar_flutter_plugin không stable như ARCore native
- **Impact**: MEDIUM - AR features có thể buggy
- **Evidence**: 
  - Plugin có ít stars trên pub.dev
  - Documentation không đầy đủ
  - Một số features chỉ work trên specific devices
- **Mitigation**:
  - ✅ Đơn giản hóa AR scope (plane detection only)
  - ✅ Test sớm trên Xiaomi 13 Pro
  - ⚠️ Fallback: Native Android nếu Flutter plugin fail

#### W5: Network dependency
- **Mô tả**: Cần WiFi stable cho phone ↔ WSL communication
- **Impact**: LOW - Nhưng có thể gây issue khi demo
- **Evidence**: 
  - Port forwarding Windows → WSL
  - Phone phải cùng WiFi network
  - Latency có thể cao (15-20s)
- **Mitigation**:
  - ✅ Test network trước khi demo
  - ✅ Có video backup nếu network fail
  - ✅ Loading animation để user không bị frustrated

### 2.3 Opportunities (Cơ hội)

#### O1: Cloud API free tiers
- **Mô tả**: Replicate, HuggingFace có free tier đủ cho development
- **Impact**: HIGH - Giải quyết VRAM limitation
- **Evidence**: 
  - Replicate: $5 free credit (~500 images)
  - HuggingFace: 1000 requests/month free
  - Latency acceptable: 5-10s
- **Exploitation**:
  - ✅ Dùng Replicate cho production quality
  - ✅ Dùng HuggingFace cho testing
  - ✅ Document cost analysis trong báo cáo

#### O2: Community resources
- **Mô tả**: Huge community cho SD, ControlNet, SAM
- **Impact**: MEDIUM - Giảm learning curve
- **Evidence**: 
  - Reddit r/StableDiffusion: 500k+ members
  - Civitai: Thousands of prompts
  - GitHub: Many example implementations
- **Exploitation**:
  - ✅ Copy best practices
  - ✅ Reuse prompts
  - ✅ Learn from others' mistakes

#### O3: Simplified AR scope vẫn impressive
- **Mô tả**: Plane detection + place object đã đủ để demo
- **Impact**: MEDIUM - Giảm risk, vẫn đạt mục tiêu
- **Evidence**: 
  - Nhiều AR apps chỉ làm basic placement
  - Giáo viên không expect production-level AR
  - Focus vào AI, AR chỉ là bonus
- **Exploitation**:
  - ✅ Allocate ít thời gian cho AR (10%)
  - ✅ Focus vào AI quality (90%)
  - ✅ AR là "cherry on top"

#### O4: Flutter hot reload
- **Mô tả**: Hot reload giúp iterate UI nhanh
- **Impact**: LOW-MEDIUM - Tăng productivity
- **Evidence**: 
  - Thay đổi UI trong <1s
  - Không cần rebuild app
- **Exploitation**:
  - ✅ Rapid prototyping UI
  - ✅ Quick bug fixes

#### O5: Academic context
- **Mô tả**: Đây là project học tập, không phải production
- **Impact**: MEDIUM - Expectations hợp lý hơn
- **Evidence**: 
  - Giáo viên hiểu limitations
  - Focus vào concept và theory
  - Demo quality > production quality
- **Exploitation**:
  - ✅ Có thể dùng workarounds
  - ✅ Document limitations openly
  - ✅ Focus vào learning outcomes

### 2.4 Threats (Rủi ro)

#### T1: Cloud API downtime/rate limits
- **Mô tả**: Replicate/HuggingFace có thể down hoặc rate limit
- **Impact**: HIGH - Block development/demo
- **Probability**: LOW-MEDIUM
- **Evidence**: 
  - APIs thỉnh thoảng maintenance
  - Free tier có rate limits
  - Network issues
- **Contingency**:
  - ✅ Test API stability sớm (Tuần 2)
  - ✅ Có 2 API providers (Replicate + HuggingFace)
  - ✅ Fallback: OpenCV inpainting (quality thấp hơn)
  - ✅ Cache results để demo offline

#### T2: Prompt engineering không đạt quality
- **Mô tả**: SD output có thể bị artifacts, không realistic
- **Impact**: MEDIUM - Ảnh hưởng demo quality
- **Probability**: MEDIUM-HIGH
- **Evidence**: 
  - Diffusion models khó control
  - Interior design cần consistency cao
  - Lighting, perspective phải match
- **Contingency**:
  - ✅ Iterate prompts nhiều lần (Tuần 2-3)
  - ✅ Dùng negative prompts
  - ✅ Test với nhiều ảnh khác nhau
  - ✅ Cherry-pick best results cho demo

#### T3: SAM mask quality không tốt
- **Mô tả**: SAM có thể segment sai, mask không clean
- **Impact**: MEDIUM - Ảnh hưởng inpainting quality
- **Probability**: MEDIUM
- **Evidence**: 
  - SAM đôi khi over-segment hoặc under-segment
  - Furniture có texture phức tạp
  - Shadows, reflections gây confusion
- **Contingency**:
  - ✅ Cho phép user adjust mask (manual refinement)
  - ✅ Morphological operations (dilate/erode) để clean mask
  - ✅ Test với nhiều loại furniture

#### T4: AR không work trên Xiaomi 13 Pro
- **Mô tả**: ar_flutter_plugin có thể không compatible
- **Impact**: MEDIUM - Mất AR feature
- **Probability**: LOW-MEDIUM
- **Evidence**: 
  - Xiaomi 13 Pro support ARCore (theo specs)
  - Nhưng Flutter plugin có thể có bugs
- **Contingency**:
  - ✅ Test AR sớm (Tuần 3 đầu)
  - ✅ Fallback: Native Android AR nếu cần
  - ✅ Worst case: Demo AR trên emulator hoặc video

#### T5: Integration issues giữa components
- **Mô tả**: SAM + Diffusion + ControlNet có thể không work seamlessly
- **Impact**: HIGH - Block entire pipeline
- **Probability**: MEDIUM
- **Evidence**: 
  - Nhiều moving parts
  - Format conversion (PIL, numpy, tensor)
  - Async communication Flutter ↔ Backend
- **Contingency**:
  - ✅ Integration testing sớm (end of Tuần 2)
  - ✅ Mock APIs để test riêng từng component
  - ✅ Comprehensive error handling

#### T6: Time overrun
- **Mô tá**: Một task nào đó mất nhiều thời gian hơn dự kiến
- **Impact**: HIGH - Delay toàn bộ timeline
- **Probability**: HIGH (realistic)
- **Evidence**: 
  - 4 tuần rất tight
  - Debugging AI models unpredictable
  - Learning curve
- **Contingency**:
  - ✅ Buffer time trong mỗi tuần
  - ✅ Parallel work khi có thể
  - ✅ Cut features nếu cần (AR first to cut)
  - ✅ MVP mindset: Core features > nice-to-haves

## 3. Risk Matrix

| Risk | Impact | Probability | Priority | Mitigation Status |
|------|--------|-------------|----------|-------------------|
| T1: API downtime | HIGH | LOW-MED | HIGH | ✅ Multiple providers |
| T2: Prompt quality | MED | MED-HIGH | HIGH | 🟡 Needs iteration |
| T3: SAM mask quality | MED | MEDIUM | MEDIUM | ✅ Manual refinement |
| T4: AR compatibility | MED | LOW-MED | LOW | 🟡 Test early |
| T5: Integration issues | HIGH | MEDIUM | HIGH | ✅ Early testing |
| T6: Time overrun | HIGH | HIGH | CRITICAL | ✅ MVP scope |

## 4. Strategic Decisions Based on SWOT

### 4.1 Leverage Strengths
1. **Use existing infrastructure** → No time wasted on setup
2. **Emphasize CV theory** → Strong academic presentation
3. **Highlight USP** → Clear differentiation in report

### 4.2 Address Weaknesses
1. **W1 (VRAM)** → Cloud API strategy (Replicate primary, HuggingFace backup)
2. **W2 (Time)** → Simplified AR scope, parallel development
3. **W3 (Experience)** → Use APIs, copy community prompts
4. **W4 (AR Flutter)** → Test early, have native fallback
5. **W5 (Network)** → Test setup, have video backup

### 4.3 Exploit Opportunities
1. **O1 (Free APIs)** → Use Replicate + HuggingFace
2. **O2 (Community)** → Reuse prompts and best practices
3. **O3 (Simple AR)** → 10% effort, 30% wow factor
4. **O5 (Academic)** → Focus on learning, not perfection

### 4.4 Mitigate Threats
1. **T1 (API)** → Multiple providers + cache + fallback
2. **T2 (Prompts)** → Iterate early, test extensively
3. **T3 (SAM)** → Manual refinement option
4. **T4 (AR)** → Test early, native fallback
5. **T5 (Integration)** → Mock testing, early integration
6. **T6 (Time)** → MVP scope, cut AR if needed

## 5. Success Criteria (Revised)

### 5.1 Must Have (MVP)
- ✅ User upload ảnh phòng
- ✅ Click để chọn vật thể (SAM segmentation)
- ✅ Backend xóa vật thể (inpainting)
- ✅ Backend generate design mới (ControlNet)
- ✅ Flutter hiển thị kết quả
- ✅ Báo cáo đầy đủ với CV theory

### 5.2 Should Have
- 🎯 AR plane detection + place 1 object
- 🎯 Style selection (2-3 options)
- 🎯 Loading animation với progress
- 🎯 Video demo chất lượng

### 5.3 Nice to Have (Stretch Goals)
- 💡 Multiple object selection
- 💡 AR với 5+ models
- 💡 Drag/rotate objects in AR
- 💡 Save/load history

## 6. Go/No-Go Decision Points

### Week 1 End: SAM Checkpoint
- **Go criteria**: SAM segmentation works với click points, mask quality acceptable
- **No-go**: Pivot to simpler segmentation (GrabCut, manual mask)

### Week 2 End: Inpainting Checkpoint
- **Go criteria**: End-to-end pipeline works (upload → segment → inpaint → result)
- **No-go**: Use OpenCV inpainting instead of SD

### Week 3 End: ControlNet Checkpoint
- **Go criteria**: Can generate 2+ design styles với acceptable quality
- **No-go**: Skip ControlNet, use simple style transfer

### Week 4 Start: AR Checkpoint
- **Go criteria**: Có đủ thời gian (3+ days) cho AR
- **No-go**: Skip AR, focus on polishing AI pipeline + report

## 7. Next Steps

Sau khi review SWOT này, chúng ta sẽ:
1. ✅ Confirm strategic decisions
2. ⏭️ Write detailed requirements cho từng tuần
3. ⏭️ Create design document với architecture
4. ⏭️ Generate task list với estimates

---

**Approval Required**: Bạn có đồng ý với SWOT analysis và strategic decisions này không? Có điểm nào cần điều chỉnh trước khi viết requirements chi tiết?
