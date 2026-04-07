# SAM3 Migration Design - Correctness Properties & Testing

This document continues from design.md with the Correctness Properties, Error Handling, and Testing Strategy sections.

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

After analyzing all acceptance criteria, I've identified the following testable properties. Note that some criteria relate to UI aesthetics or test coverage goals and are not included as they're not computationally verifiable.

### Property 1: Initialization Correctness

*For any* SAM3Segmentation instance, after initialization, the model ID shall be set to the correct Replicate model reference.

**Validates: Requirements 1.1**

### Property 2: Text Segmentation Output Format

*For any* valid image (numpy array) and non-empty text prompt, calling segment_by_text() shall return a tuple of (mask, confidence) where mask is a binary numpy array with the same height and width as the input image, and confidence is a float between 0.0 and 1.0.

**Validates: Requirements 1.2**

### Property 3: Point Segmentation Output Format

*For any* valid image and list of point coordinates with matching labels, calling segment_by_points() shall return a tuple of (mask, confidence) where mask is a binary numpy array matching input dimensions, and confidence is between 0.0 and 1.0.

**Validates: Requirements 1.3**

### Property 4: Segmentation Logging

*For any* successful segmentation operation (text or points), the system logs shall contain an entry with cost ($0.00098) and processing time in milliseconds.

**Validates: Requirements 1.5**

### Property 5: Text API Response Contract

*For any* valid image_id and non-empty text_prompt, POST to /api/v1/segmentation/sam3/text shall return HTTP 200 with a JSON response containing fields: status, mask_id, mask_url, confidence, processing_time_ms, and cost_usd.

**Validates: Requirements 2.1, 2.4**

### Property 6: Whitespace Input Rejection

*For any* string composed entirely of whitespace characters (spaces, tabs, newlines, or empty), POST to /api/v1/segmentation/sam3/text shall return HTTP 400 Bad Request.

**Validates: Requirements 2.2, 5.3**

### Property 7: Missing Resource Error Handling

*For any* non-existent image_id, POST to /api/v1/segmentation/sam3/text or /api/v1/segmentation/sam3/points shall return HTTP 404 Not Found.

**Validates: Requirements 2.3**

### Property 8: Point API Response Contract

*For any* valid image_id and non-empty points array, POST to /api/v1/segmentation/sam3/points shall return HTTP 200 with a JSON response containing mask_id.

**Validates: Requirements 3.1**

### Property 9: Out-of-Bounds Point Rejection

*For any* point coordinate where x >= image_width or y >= image_height or x < 0 or y < 0, POST to /api/v1/segmentation/sam3/points shall return HTTP 400 Bad Request with a validation error message.

**Validates: Requirements 3.3**

### Property 10: Mask Persistence Round-Trip

*For any* successful segmentation operation, the returned mask_id shall allow retrieval of the mask from storage, and the retrieved mask shall have the same dimensions and non-zero pixel count as the original.

**Validates: Requirements 3.5, 13.3**

### Property 11: Mode Switch State Reset

*For any* UI state with active selection (text or points), switching segmentation modes shall clear both the selection data and the displayed mask.

**Validates: Requirements 4.4, 4.5**

### Property 12: Button Enable State

*For any* text input state, the "Segment" button shall be enabled if and only if the text is non-empty and not composed entirely of whitespace.

**Validates: Requirements 5.1**

### Property 13: Point Marker Visibility

*For any* list of N points added to the image, the UI shall display exactly N numbered markers at the corresponding coordinates.

**Validates: Requirements 7.3**

### Property 14: Mask-Image Alignment

*For any* displayed mask overlay, the mask dimensions shall match the image dimensions, and zoom/pan transformations applied to the image shall be applied identically to the mask.

**Validates: Requirements 8.4, 8.5**

### Property 15: Error State Clears Loading

*For any* error condition (network, API, validation), when the error message is displayed, the loading indicator shall be hidden.

**Validates: Requirements 9.5**

### Property 16: Metrics Recording

*For any* segmentation request, the system shall record a timestamp at request start, and upon completion, shall calculate and log processing_time_ms as the difference between completion and start timestamps.

**Validates: Requirements 10.1, 10.2, 10.3**

### Property 17: API Version Indication

*For any* successful API response from segmentation endpoints, the response shall include a field indicating which SAM version (v1 or v3) was used to process the request.

**Validates: Requirements 11.5**

### Property 18: Mask Metadata Persistence

*For any* saved mask, the metadata file shall contain fields for timestamp, image_id, and segmentation_type, and these values shall be retrievable using the mask_id.

**Validates: Requirements 13.2**

### Property 19: Mask Format Consistency

*For any* mask retrieved from storage using a mask_id, the returned data shall be a numpy array with dtype uint8 and values in the set {0, 255}.

**Validates: Requirements 13.4**

### Property 20: Most Recent Mask Selection

*For any* image_id with multiple associated masks, when the inpainting screen loads, the system shall select the mask with the most recent timestamp.

**Validates: Requirements 14.5**

---

## Error Handling

### Error Categories

#### 1. Input Validation Errors (HTTP 400)

**Scenarios**:
- Empty or whitespace-only text prompts
- Empty points array
- Point coordinates out of image bounds
- Invalid image_id format
- Missing required fields in request body

**Handling Strategy**:
```python
# Validate before expensive operations
if not text_prompt or text_prompt.isspace():
    raise HTTPException(
        status_code=400,
        detail={
            "error_type": "ValidationError",
            "message": "Text prompt cannot be empty or whitespace-only",
            "field": "text_prompt"
        }
    )
```

**User-Facing Messages**:
- "Please enter a description of the objects to segment"
- "Please tap on the image to select objects"
- "Point is outside image boundaries"

#### 2. Resource Not Found Errors (HTTP 404)

**Scenarios**:
- image_id does not exist in storage
- mask_id does not exist in storage
- Uploaded image file was deleted

**Handling Strategy**:
```python
if not image_path.exists():
    raise HTTPException(
        status_code=404,
        detail={
            "error_type": "ResourceNotFound",
            "message": f"Image with ID '{image_id}' not found",
            "suggestion": "Please upload the image again"
        }
    )
```

**User-Facing Messages**:
- "Image not found. Please upload again."
- "Mask not found. Please segment the image again."

#### 3. External Service Errors (HTTP 503)

**Scenarios**:
- Replicate API is down or unreachable
- Network timeout (>30 seconds)
- API rate limit exceeded
- Invalid API token

**Handling Strategy**:
```python
try:
    output = replicate.run(self.model, input={...})
except replicate.exceptions.ReplicateError as e:
    logger.error(f"Replicate API error: {str(e)}")
    raise HTTPException(
        status_code=503,
        detail={
            "error_type": "ServiceUnavailable",
            "message": "Segmentation service temporarily unavailable",
            "suggestion": "Please try again in a few moments"
        }
    )
```

**Retry Logic**:
- Retry up to 3 times with exponential backoff (1s, 2s, 4s)
- Only retry on network errors, not on validation errors
- Log each retry attempt

**User-Facing Messages**:
- "Service temporarily unavailable. Please try again later."
- "Network error. Please check your connection."

#### 4. Processing Errors (HTTP 500)

**Scenarios**:
- Mask download fails (URL invalid or expired)
- Image encoding/decoding fails
- Unexpected API response format
- Disk full (cannot save mask)

**Handling Strategy**:
```python
try:
    mask = self._download_mask(mask_url)
except Exception as e:
    logger.error(f"Failed to download mask: {str(e)}", exc_info=True)
    raise RuntimeError(
        f"Failed to process segmentation result: {str(e)}"
    )
```

**User-Facing Messages**:
- "An unexpected error occurred. Please try again."
- "Failed to save segmentation result. Please check storage space."

### Error Logging Strategy

**Log Levels**:
- **ERROR**: API failures, unexpected exceptions, data corruption
- **WARNING**: Slow requests (>15s), high API usage (>1000/day), retry attempts
- **INFO**: Successful operations, cost tracking, performance metrics
- **DEBUG**: Request/response details, intermediate processing steps

**Log Format**:
```python
logger.error(
    f"SAM3 segmentation failed",
    extra={
        "image_id": image_id,
        "mode": "text" or "points",
        "error_type": type(e).__name__,
        "processing_time_ms": elapsed_ms,
        "retry_count": retry_count
    }
)
```

### Graceful Degradation

**No Fallback to SAM v1**: Unlike the inpainting service, SAM3 does NOT fall back to SAM v1 on failure. This is intentional:
- SAM v1 requires 4GB VRAM (defeats the purpose of migration)
- Different API contracts (SAM v1 requires pre-loaded image)
- Clear error messages guide users to retry

**Timeout Handling**:
- Set 30-second timeout for Replicate API calls
- If timeout occurs, return 503 with clear message
- Log timeout events for monitoring

---

## Testing Strategy

### Dual Testing Approach

This feature requires both **unit tests** and **property-based tests** for comprehensive coverage:

- **Unit tests**: Verify specific examples, edge cases, and error conditions
- **Property tests**: Verify universal properties across randomized inputs
- Together they provide confidence in correctness and catch regressions

### Backend Testing

#### Unit Tests (pytest)

**Location**: `backend/tests/test_sam3_segmentation.py`

**Test Cases**:
```python
def test_sam3_initialization():
    """Verify SAM3 initializes with correct model ID"""
    sam3 = SAM3Segmentation()
    assert sam3.model == "mattsays/sam3-image:d73db077..."

def test_segment_by_text_empty_prompt():
    """Verify empty text prompt raises ValueError"""
    sam3 = SAM3Segmentation()
    with pytest.raises(ValueError, match="empty"):
        sam3.segment_by_text(image, "")

def test_segment_by_points_empty_array():
    """Verify empty points array raises ValueError"""
    sam3 = SAM3Segmentation()
    with pytest.raises(ValueError, match="empty"):
        sam3.segment_by_points(image, [], [])

def test_api_retry_logic(mock_replicate):
    """Verify API retries 3 times on failure"""
    mock_replicate.run.side_effect = [
        Exception("Network error"),
        Exception("Network error"),
        {"output": "https://mask.png"}
    ]
    sam3 = SAM3Segmentation()
    mask, conf = sam3.segment_by_text(image, "sofa")
    assert mock_replicate.run.call_count == 3

def test_mask_download_invalid_url():
    """Verify invalid mask URL raises RuntimeError"""
    sam3 = SAM3Segmentation()
    with pytest.raises(RuntimeError, match="download"):
        sam3._download_mask("https://invalid.url/mask.png")
```

#### Property-Based Tests (Hypothesis)

**Configuration**: Minimum 100 iterations per test

**Test Cases**:
```python
from hypothesis import given, strategies as st
import numpy as np

@given(
    height=st.integers(min_value=100, max_value=2000),
    width=st.integers(min_value=100, max_value=2000),
    text=st.text(min_size=1, max_size=100)
)
def test_property_text_segmentation_output_format(height, width, text):
    """
    Feature: sam3-dual-mode-segmentation, Property 2: Text Segmentation Output Format
    For any valid image and non-empty text, output is correctly formatted mask
    """
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    sam3 = SAM3Segmentation()
    
    mask, confidence = sam3.segment_by_text(image, text)
    
    # Verify output format
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (height, width)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
    assert 0.0 <= confidence <= 1.0

@given(whitespace=st.text(alphabet=' \t\n\r', min_size=0, max_size=50))
def test_property_whitespace_rejection(whitespace):
    """
    Feature: sam3-dual-mode-segmentation, Property 6: Whitespace Input Rejection
    For any whitespace-only string, API returns 400
    """
    response = client.post(
        "/api/v1/segmentation/sam3/text",
        json={"image_id": "test_img", "text_prompt": whitespace}
    )
    assert response.status_code == 400
```

#### API Integration Tests

**Location**: `backend/tests/test_segmentation_api.py`

**Test Cases**:
```python
def test_text_segmentation_api_success():
    """Verify text API returns correct response structure"""
    response = client.post(
        "/api/v1/segmentation/sam3/text",
        json={"image_id": valid_image_id, "text_prompt": "sofa, chair"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "mask_id" in data
    assert "mask_url" in data
    assert "confidence" in data
    assert "processing_time_ms" in data
    assert "cost_usd" in data

def test_mask_persistence_round_trip():
    """Verify saved masks can be retrieved"""
    # Segment and get mask_id
    response = client.post(
        "/api/v1/segmentation/sam3/text",
        json={"image_id": valid_image_id, "text_prompt": "table"}
    )
    mask_id = response.json()["mask_id"]
    
    # Retrieve mask
    mask = MaskStorage.load_mask(mask_id)
    assert mask is not None
    assert mask.shape == (expected_height, expected_width)
```

### Frontend Testing (Flutter)

#### Widget Tests

**Location**: `frontend/test/segmentation_screen_test.dart`

**Test Cases**:
```dart
testWidgets('Mode selector displays Text and Tap options', (tester) async {
  await tester.pumpWidget(MyApp());
  await tester.tap(find.text('Segment'));
  await tester.pumpAndSettle();
  
  expect(find.text('Text'), findsOneWidget);
  expect(find.text('Tap'), findsOneWidget);
});

testWidgets('Segment button disabled when text empty', (tester) async {
  await tester.pumpWidget(MyApp());
  await tester.tap(find.text('Text'));
  await tester.pumpAndSettle();
  
  final button = find.widgetWithText(ElevatedButton, 'Segment');
  expect(tester.widget<ElevatedButton>(button).enabled, isFalse);
  
  await tester.enterText(find.byType(TextField), 'sofa');
  await tester.pump();
  
  expect(tester.widget<ElevatedButton>(button).enabled, isTrue);
});
```

### Test Coverage Goals

- **Backend Core Logic**: >90% coverage
- **API Endpoints**: >85% coverage
- **Flutter Widgets**: >80% coverage
- **Integration Tests**: Cover all critical user flows

### Performance Benchmarks

**Target Metrics**:
- Text segmentation: < 5 seconds (95th percentile)
- Point segmentation: < 4 seconds (95th percentile)
- Mask download: < 1 second
- Total API response time: < 6 seconds

---

## Summary

This design document provides a comprehensive technical specification for migrating from SAM v1 to SAM3 for the AI Interior Design project. The migration addresses the critical VRAM constraint (GTX 1650 4GB) by moving segmentation to the cloud, enabling the system to focus GPU resources on inpainting and generation tasks.

**Key Benefits**:
- Zero VRAM usage for segmentation
- Dual-mode support (text + points) for better UX
- Faster inference (~2-5 seconds vs 10+ seconds)
- Simplified deployment and maintenance
- Cost-effective ($0.00098 per request)

**Implementation Priority**: This is the highest priority feature (Mục tiêu 2: Semantic Inpainting) and should be completed in Week 1 of the 4-week roadmap.

**Next Steps**: Proceed to implementation following the phased migration strategy outlined in the Integration Strategy section.
