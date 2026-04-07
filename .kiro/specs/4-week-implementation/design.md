# Design Document: 4-Week AI Interior Design Implementation

## 1. System Architecture

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Mobile App (Flutter)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Camera       │  │ Segmentation │  │ AR View      │      │
│  │ Screen       │  │ Screen       │  │ Screen       │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
│         └──────────────────┴──────────────────┘              │
│                            │                                 │
│                   ┌────────▼────────┐                        │
│                   │  API Service    │                        │
│                   │  (HTTP Client)  │                        │
│                   └────────┬────────┘                        │
└────────────────────────────┼─────────────────────────────────┘
                             │ HTTP/REST
                             │ (WiFi: 192.168.1.12:8000)
                             │
┌────────────────────────────▼─────────────────────────────────┐
│              Backend Server (FastAPI on WSL)                 │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                   API Layer (v1)                      │  │
│  │  /segmentation  /inpainting  /generation  /health    │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │              Core Business Logic                      │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │ SAM         │  │ Diffusion   │  │ ControlNet  │  │  │
│  │  │ Segmentation│  │ Inpainting  │  │ Generation  │  │  │
│  │  │ (Local)     │  │ (Cloud API) │  │ (Cloud API) │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │              External Services                        │  │
│  │  ┌─────────────────┐      ┌─────────────────┐        │  │
│  │  │ Replicate API   │      │ HuggingFace API │        │  │
│  │  │ (Primary)       │      │ (Fallback)      │        │  │
│  │  └─────────────────┘      └─────────────────┘        │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  Data Storage                          │  │
│  │  inputs/  outputs/  masks/  temp/                     │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  GTX 1650 4GB  │
                    │  (CUDA)        │
                    │  - SAM only    │
                    └────────────────┘
```

### 1.2 Data Flow

#### Flow 1: Image Segmentation (Week 1-2)
```
User → Upload Image → Flutter
         │
         ▼
    HTTP POST /api/v1/segmentation/segment
         │
         ▼
    Backend: Load image, resize if needed
         │
         ▼
    SAM: Set image (local GPU)
         │
         ▼
    Return: image_id, ready status
         │
         ▼
    Flutter: Display image, enable point selection
         │
         ▼
User → Click points on objects
         │
         ▼
    HTTP POST /api/v1/segmentation/segment-with-points
    Body: {image_id, points: [{x, y, label}]}
         │
         ▼
    Backend: SAM predict with points (local GPU)
         │
         ▼
    Generate mask, save to masks/
         │
         ▼
    Return: mask_url, confidence
         │
         ▼
    Flutter: Display mask overlay
```

#### Flow 2: Inpainting (Week 2)
```
User → Confirm mask → Flutter
         │
         ▼
    HTTP POST /api/v1/inpainting/remove-object
    Body: {image_id, mask_id}
         │
         ▼
    Backend: Load image + mask
         │
         ▼
    Prepare for inpainting (convert formats)
         │
         ▼
    Call Replicate API: SD Inpainting
    Prompt: "empty room, clean floor, white walls, photorealistic"
    Negative: "furniture, objects, clutter"
         │
         ▼ (5-10s latency)
    Replicate returns inpainted image
         │
         ▼
    Save to outputs/, return URL
         │
         ▼
    Flutter: Display "empty room" result
```

#### Flow 3: Design Generation (Week 3)
```
User → Select style → Flutter
         │
         ▼
    HTTP POST /api/v1/generation/generate-design
    Body: {inpainted_image_id, style: "modern"}
         │
         ▼
    Backend: Load inpainted image
         │
         ▼
    Canny edge detection (OpenCV, local)
         │
         ▼
    Call Replicate API: ControlNet + SD
    Prompt: "modern interior design, minimalist furniture, ..."
    ControlNet: Canny edges
         │
         ▼ (10-15s latency)
    Replicate returns styled image
         │
         ▼
    Save to outputs/, return URL
         │
         ▼
    Flutter: Display new design
```

#### Flow 4: AR Placement (Week 4)
```
User → Enter AR mode → Flutter
         │
         ▼
    ARCore: Initialize session
         │
         ▼
    Detect horizontal planes
         │
         ▼
    Display plane visualization (grid)
         │
         ▼
User → Tap on plane
         │
         ▼
    Load 3D model (.glb)
         │
         ▼
    Place at tap position
         │
         ▼
    Render in AR view
```

## 2. Component Design

### 2.1 Backend Components

#### 2.1.1 SAM Segmentation Module
**File**: `backend/app/core/sam_segmentation.py`

**Class**: `SAMSegmentation`

**Methods**:
```python
def set_image(image: np.ndarray) -> None
    """Set image for segmentation"""
    
def segment_by_points(
    point_coords: List[Tuple[int, int]],
    point_labels: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
    """Segment using point prompts"""
    # Returns: (masks, scores, logits)
    
def get_best_mask(
    masks: np.ndarray,
    scores: np.ndarray
) -> np.ndarray
    """Get highest confidence mask"""
```

**Performance**:
- Inference time: ~200-500ms (GTX 1650)
- VRAM usage: ~2GB
- Input: RGB image (H, W, 3)
- Output: Binary mask (H, W)

#### 2.1.2 Diffusion Inpainting Module
**File**: `backend/app/core/diffusion_inpainting.py`

**Class**: `DiffusionInpainting`

**Methods**:
```python
def inpaint_with_replicate(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str = None
) -> Image.Image
    """Inpaint using Replicate API"""
    
def inpaint_with_huggingface(
    image: Image.Image,
    mask: Image.Image,
    prompt: str
) -> Image.Image
    """Fallback: HuggingFace API"""
    
def inpaint_with_opencv(
    image: np.ndarray,
    mask: np.ndarray
) -> np.ndarray
    """Emergency fallback: OpenCV inpainting"""
```

**Prompts** (Optimized):
```python
INPAINTING_PROMPTS = {
    "empty_room": {
        "positive": "empty room, clean floor, white walls, natural lighting, photorealistic, high quality, 8k",
        "negative": "furniture, objects, clutter, people, text, watermark, blurry, low quality"
    }
}
```

**Performance**:
- Replicate latency: 5-10s
- HuggingFace latency: 8-15s
- OpenCV latency: <1s (but lower quality)

#### 2.1.3 ControlNet Generation Module
**File**: `backend/app/core/controlnet_generation.py`

**Class**: `ControlNetGeneration`

**Methods**:
```python
def detect_edges(image: np.ndarray) -> np.ndarray
    """Canny edge detection"""
    
def generate_with_controlnet(
    image: Image.Image,
    edges: Image.Image,
    style: str
) -> Image.Image
    """Generate design with ControlNet"""
```

**Style Prompts**:
```python
STYLE_PROMPTS = {
    "modern": {
        "positive": "modern interior design, minimalist furniture, clean lines, neutral colors, natural light, scandinavian style, high quality",
        "negative": "cluttered, baroque, ornate, dark, old-fashioned"
    },
    "minimalist": {
        "positive": "minimalist interior, white walls, simple furniture, open space, natural materials, zen, peaceful",
        "negative": "busy, colorful, decorative, cluttered"
    },
    "industrial": {
        "positive": "industrial interior, exposed brick, metal furniture, concrete floor, loft style, urban",
        "negative": "traditional, soft, pastel, ornate"
    }
}
```

#### 2.1.4 API Endpoints

**Segmentation Endpoints**:
```python
POST /api/v1/segmentation/upload
    Request: multipart/form-data (image file)
    Response: {image_id, image_shape, status}
    
POST /api/v1/segmentation/segment-points
    Request: {image_id, points: [{x, y, label}]}
    Response: {mask_id, mask_url, confidence}
```

**Inpainting Endpoints**:
```python
POST /api/v1/inpainting/remove-object
    Request: {image_id, mask_id}
    Response: {result_id, result_url, processing_time}
```

**Generation Endpoints**:
```python
POST /api/v1/generation/generate-design
    Request: {image_id, style}
    Response: {design_id, design_url, processing_time}
    
GET /api/v1/generation/styles
    Response: {styles: ["modern", "minimalist", "industrial"]}
```

### 2.2 Frontend Components

#### 2.2.1 Screen Structure
```
lib/
├── screens/
│   ├── home_screen.dart           # Entry point
│   ├── camera_screen.dart         # Capture/select image
│   ├── segmentation_screen.dart   # Point selection
│   ├── inpainting_screen.dart     # View inpainted result
│   ├── generation_screen.dart     # Style selection + result
│   └── ar_screen.dart             # AR placement
```

#### 2.2.2 Service Layer
```dart
// lib/services/api_service.dart
class ApiService {
  Future<UploadResponse> uploadImage(File image);
  Future<SegmentResponse> segmentWithPoints(String imageId, List<Point> points);
  Future<InpaintResponse> removeObject(String imageId, String maskId);
  Future<GenerateResponse> generateDesign(String imageId, String style);
}

// lib/services/image_service.dart
class ImageService {
  Future<File?> pickFromCamera();
  Future<File?> pickFromGallery();
  Future<File> compressImage(File image);
}

// lib/services/ar_service.dart
class ARService {
  Future<void> initializeAR();
  Future<List<ARPlane>> detectPlanes();
  void placeObject(ARPlane plane, String modelPath);
}
```

#### 2.2.3 State Management
```dart
// Using Provider pattern
class AppState extends ChangeNotifier {
  File? currentImage;
  String? imageId;
  List<Point> selectedPoints = [];
  String? maskId;
  String? inpaintedImageUrl;
  String? generatedDesignUrl;
  
  ProcessingStatus status = ProcessingStatus.idle;
  
  void setImage(File image) { ... }
  void addPoint(Point point) { ... }
  void setProcessing(ProcessingStatus status) { ... }
}
```

### 2.3 AR Component Design

#### 2.3.1 Simplified AR Architecture
```dart
// lib/screens/ar_screen.dart
class ARScreen extends StatefulWidget {
  @override
  _ARScreenState createState() => _ARScreenState();
}

class _ARScreenState extends State<ARScreen> {
  ARSessionManager? arSessionManager;
  List<ARPlane> detectedPlanes = [];
  ARNode? placedObject;
  
  @override
  void initState() {
    super.initState();
    _initializeAR();
  }
  
  void _initializeAR() async {
    arSessionManager = await ARSessionManager.create();
    arSessionManager!.onPlaneDetected = _onPlaneDetected;
  }
  
  void _onPlaneDetected(ARPlane plane) {
    setState(() {
      detectedPlanes.add(plane);
    });
  }
  
  void _onPlaneTap(ARPlane plane) {
    // Load 3D model
    final model = await loadGLB('assets/models/chair.glb');
    
    // Place at plane center
    placedObject = ARNode(
      model: model,
      position: plane.centerPose.position,
      rotation: plane.centerPose.rotation,
    );
    
    arSessionManager!.addNode(placedObject!);
  }
}
```

#### 2.3.2 3D Models
```
assets/
└── models/
    ├── chair.glb       # Simple chair model
    ├── table.glb       # Simple table model
    └── sofa.glb        # Simple sofa model
```

**Model Requirements**:
- Format: GLB (binary glTF)
- Size: <5MB each
- Polygons: <10k triangles
- Textures: Embedded, <1024x1024

## 3. Database Schema (File-based)

### 3.1 Data Storage Structure
```
backend/data/
├── inputs/
│   └── {timestamp}_{uuid}.jpg          # Original images
├── masks/
│   └── {image_id}_{mask_id}.png        # Generated masks
├── outputs/
│   ├── inpainted/
│   │   └── {image_id}_inpainted.jpg    # Inpainted results
│   └── generated/
│       └── {image_id}_{style}.jpg      # Generated designs
└── temp/
    └── {uuid}.tmp                       # Temporary files
```

### 3.2 Metadata (JSON)
```json
// backend/data/metadata.json
{
  "images": {
    "image_123": {
      "original_path": "inputs/20240224_123456_uuid.jpg",
      "upload_time": "2024-02-24T12:34:56Z",
      "dimensions": {"width": 1920, "height": 1080},
      "masks": ["mask_1", "mask_2"]
    }
  },
  "masks": {
    "mask_1": {
      "image_id": "image_123",
      "path": "masks/image_123_mask_1.png",
      "confidence": 0.95,
      "points": [{"x": 100, "y": 150, "label": 1}]
    }
  },
  "results": {
    "result_1": {
      "type": "inpainted",
      "image_id": "image_123",
      "mask_id": "mask_1",
      "path": "outputs/inpainted/image_123_inpainted.jpg",
      "processing_time": 8.5
    }
  }
}
```

## 4. Error Handling Strategy

### 4.1 Backend Error Handling
```python
# app/api/v1/endpoints/segmentation.py
@router.post("/segment-points")
async def segment_with_points(...):
    try:
        # Main logic
        result = sam_seg.segment_by_points(...)
        return {"status": "success", "data": result}
        
    except ValueError as e:
        # Invalid input
        raise HTTPException(status_code=400, detail=str(e))
        
    except torch.cuda.OutOfMemoryError:
        # VRAM issue
        torch.cuda.empty_cache()
        raise HTTPException(status_code=503, detail="GPU memory full, try again")
        
    except Exception as e:
        # Unexpected error
        logger.error(f"Segmentation error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

### 4.2 API Client Error Handling
```python
# app/services/replicate_client.py
async def inpaint_with_retry(image, mask, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = await replicate.run(...)
            return result
            
        except replicate.exceptions.RateLimitError:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                # Fallback to HuggingFace
                return await huggingface_client.inpaint(...)
                
        except replicate.exceptions.APIError as e:
            logger.error(f"Replicate API error: {e}")
            # Fallback to HuggingFace
            return await huggingface_client.inpaint(...)
```

### 4.3 Flutter Error Handling
```dart
// lib/services/api_service.dart
Future<SegmentResponse> segmentWithPoints(...) async {
  try {
    final response = await http.post(...).timeout(Duration(seconds: 30));
    
    if (response.statusCode == 200) {
      return SegmentResponse.fromJson(jsonDecode(response.body));
    } else if (response.statusCode == 503) {
      throw ServerBusyException("Server is busy, please try again");
    } else {
      throw ApiException("Failed to segment: ${response.statusCode}");
    }
    
  } on TimeoutException {
    throw NetworkException("Request timeout, check your connection");
    
  } on SocketException {
    throw NetworkException("Cannot connect to server");
    
  } catch (e) {
    throw ApiException("Unexpected error: $e");
  }
}
```

## 5. Performance Optimization

### 5.1 Backend Optimizations
```python
# 1. Model caching (already implemented in dependencies.py)
@lru_cache()
def get_model_manager() -> ModelManager:
    return ModelManager()

# 2. Image preprocessing
def optimize_image_for_processing(image: Image.Image) -> Image.Image:
    # Resize if too large
    if max(image.size) > 2048:
        image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    return image

# 3. CUDA memory management
with torch.inference_mode():
    masks, scores, logits = predictor.predict(...)
torch.cuda.empty_cache()

# 4. Async API calls
async def process_pipeline(image_id, mask_id):
    # Don't block on API calls
    result = await asyncio.create_task(
        replicate_client.inpaint(...)
    )
    return result
```

### 5.2 Frontend Optimizations
```dart
// 1. Image compression before upload
Future<File> compressImage(File image) async {
  final bytes = await image.readAsBytes();
  final compressed = await FlutterImageCompress.compressWithList(
    bytes,
    quality: 85,
    minWidth: 1920,
    minHeight: 1080,
  );
  return File('${image.path}_compressed.jpg')..writeAsBytesSync(compressed);
}

// 2. Caching network images
CachedNetworkImage(
  imageUrl: resultUrl,
  placeholder: (context, url) => CircularProgressIndicator(),
  errorWidget: (context, url, error) => Icon(Icons.error),
)

// 3. Lazy loading AR models
Future<void> preloadARModels() async {
  await Future.wait([
    loadGLB('assets/models/chair.glb'),
    loadGLB('assets/models/table.glb'),
  ]);
}
```

## 6. Testing Strategy

### 6.1 Backend Testing
```python
# tests/test_sam_segmentation.py
def test_sam_segmentation():
    sam_seg = SAMSegmentation(predictor)
    image = load_test_image("room.jpg")
    sam_seg.set_image(image)
    
    points = [(100, 150), (200, 250)]
    labels = [1, 1]
    
    masks, scores, logits = sam_seg.segment_by_points(points, labels)
    
    assert len(masks) == 3  # multimask_output=True
    assert scores[0] > 0.5  # Reasonable confidence
    assert masks[0].shape == image.shape[:2]
```

### 6.2 Integration Testing
```python
# tests/test_integration.py
async def test_full_pipeline():
    # 1. Upload image
    response = await client.post("/api/v1/segmentation/upload", files={"file": image})
    image_id = response.json()["image_id"]
    
    # 2. Segment
    response = await client.post("/api/v1/segmentation/segment-points", json={
        "image_id": image_id,
        "points": [{"x": 100, "y": 150, "label": 1}]
    })
    mask_id = response.json()["mask_id"]
    
    # 3. Inpaint
    response = await client.post("/api/v1/inpainting/remove-object", json={
        "image_id": image_id,
        "mask_id": mask_id
    })
    assert response.status_code == 200
    assert "result_url" in response.json()
```

### 6.3 Manual Testing Checklist
```markdown
Week 1:
- [ ] SAM loads successfully on GTX 1650
- [ ] Can segment furniture with 2-3 clicks
- [ ] Mask quality is acceptable (>80% accuracy visually)
- [ ] API responds within 1s for segmentation

Week 2:
- [ ] Replicate API works with test images
- [ ] Inpainting removes objects cleanly
- [ ] "Empty room" looks realistic
- [ ] End-to-end latency <15s

Week 3:
- [ ] Canny edges capture room structure
- [ ] ControlNet generates coherent designs
- [ ] 3 styles produce visually different results
- [ ] Generated designs match room layout

Week 4:
- [ ] AR initializes on Xiaomi 13 Pro
- [ ] Planes detected within 5s
- [ ] Objects placed at correct position
- [ ] Objects stay stable when moving camera
```

## 7. Deployment Configuration

### 7.1 Backend Deployment
```bash
# backend/.env
DEBUG=False
DEVICE=cuda
MAX_IMAGE_SIZE=2048
REPLICATE_API_KEY=r8_xxx
HUGGINGFACE_API_KEY=hf_xxx

# Run server
cd backend
python -m app.main
# Server runs on 0.0.0.0:8000
```

### 7.2 Port Forwarding (Windows → WSL)
```powershell
# Already configured in setup_port_forward.ps1
# Run as Administrator before demo
.\setup_port_forward.ps1
```

### 7.3 Flutter Build
```bash
# Development
flutter run --release

# Production APK
flutter build apk --release
# Output: build/app/outputs/flutter-apk/app-release.apk
```

## 8. Monitoring & Logging

### 8.1 Backend Logging
```python
# app/utils/logger.py
logger.info(f"🖼️  Processing image: {image_id}")
logger.info(f"🎯 Segmentation completed in {elapsed:.2f}s")
logger.info(f"🎨 Inpainting started with Replicate")
logger.error(f"❌ API error: {error}")
```

### 8.2 Performance Metrics
```python
# Track in metadata.json
{
  "metrics": {
    "sam_inference_time": 0.35,
    "inpainting_time": 8.5,
    "generation_time": 12.3,
    "total_pipeline_time": 21.15
  }
}
```

## 9. Documentation Requirements

### 9.1 Code Documentation
- Docstrings for all public methods
- Type hints for all function signatures
- Inline comments for complex logic

### 9.2 API Documentation
- OpenAPI/Swagger auto-generated from FastAPI
- Available at: `http://localhost:8000/docs`

### 9.3 User Documentation
- README.md with setup instructions
- Video demo showing full workflow
- Troubleshooting guide

---

**Next Step**: Generate detailed tasks.md with week-by-week breakdown and time estimates.
