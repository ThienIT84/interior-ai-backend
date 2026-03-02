# Hybrid Inpainting Strategy

## Overview

Dự án sử dụng **Hybrid Inpainting** với 2 backends:

| Method | Speed | Cost | Use Case |
|--------|-------|------|----------|
| **Local GPU** | 15 min | FREE | Development, Testing |
| **Replicate API** | 10 sec | $0.01/image | Demo, Presentation |

## Why Hybrid?

**Problem**: 
- HuggingFace free tier không còn hoạt động (Error 410)
- Local GPU (GTX 1650) quá chậm (15 phút/ảnh)

**Solution**:
- Development: Dùng Local (free, unlimited)
- Demo: Dùng Replicate (fast, impressive, chỉ tốn $0.01/ảnh)

## Setup

### Option 1: Local Only (FREE)

```bash
# No setup needed!
# Just set method to "local" in .env
INPAINTING_METHOD=local
```

**Pros**:
- ✅ Completely FREE
- ✅ No API limits
- ✅ Works offline

**Cons**:
- ❌ Very slow (15 minutes/image)
- ❌ Not suitable for demo

### Option 2: Replicate API (RECOMMENDED for Demo)

```bash
# 1. Get Replicate API token
# Visit: https://replicate.com/account/api-tokens

# 2. Add to .env
REPLICATE_API_TOKEN=r8_your_token_here
INPAINTING_METHOD=replicate

# 3. Install replicate package
pip install replicate
```

**Pros**:
- ✅ Very fast (10 seconds/image)
- ✅ High quality
- ✅ Perfect for demo

**Cons**:
- ❌ Costs $0.01/image (~230 VNĐ)
- ❌ Requires internet

### Option 3: Auto (BEST)

```bash
# Set to "auto" in .env
INPAINTING_METHOD=auto
```

**Behavior**:
- If Replicate token available → Use Replicate (fast)
- If Replicate token not set → Use Local (free)
- If Replicate fails → Auto-fallback to Local

## Cost Estimation

### For Development (1 month)
- Test images: ~50 images
- Method: Local (FREE)
- Cost: **$0**

### For Demo (1 day)
- Demo images: ~10 images
- Method: Replicate
- Cost: **$0.10** (~2,300 VNĐ)

### For Presentation (1 hour)
- Live demo: ~3 images
- Method: Replicate
- Cost: **$0.03** (~700 VNĐ)

**Total estimated cost: $0.13 (~3,000 VNĐ)**

## Usage in Code

### Automatic (Recommended)

```python
from app.core.inpainting_service import get_inpainting_service

service = get_inpainting_service()  # Uses INPAINTING_METHOD from .env

result, metadata = service.inpaint(
    image=image,
    mask=mask,
    prompt="empty room, clean floor"
)

print(f"Method used: {metadata['method']}")
print(f"Time: {metadata['processing_time']:.1f}s")
print(f"Cost: ${metadata.get('cost', 0):.2f}")
```

### Force Specific Method

```python
# Force local (for testing)
result, metadata = service.inpaint(
    image=image,
    mask=mask,
    method="local"  # Override
)

# Force Replicate (for demo)
result, metadata = service.inpaint(
    image=image,
    mask=mask,
    method="replicate"  # Override
)
```

## Recommendations

### For Development
```bash
INPAINTING_METHOD=local
```
- Free, unlimited testing
- Accept slow speed

### For Demo Day
```bash
INPAINTING_METHOD=replicate
```
- Fast, impressive
- Worth the $0.01/image

### For Submission
```bash
INPAINTING_METHOD=auto
```
- Flexible
- Works with or without Replicate token

## Testing

```bash
# Test local
python test_local_inpainting.py

# Test Replicate
python test_replicate_inpainting.py

# Test hybrid (auto-select)
python test_hybrid_inpainting.py
```

## Troubleshooting

### "Replicate API token not set"
- Get token at: https://replicate.com/account/api-tokens
- Add to .env: `REPLICATE_API_TOKEN=r8_xxx`

### "Local inpainting too slow"
- This is expected (15 min on GTX 1650)
- Use Replicate for demo
- Or reduce steps: `num_inference_steps=25` (faster but lower quality)

### "Replicate API failed"
- Check internet connection
- Check API token is valid
- System will auto-fallback to local if `INPAINTING_METHOD=auto`

## Summary

**Best Strategy**:
1. Development: Use Local (free)
2. Testing: Use Local (free)
3. Demo: Use Replicate ($0.01/image)
4. Set `INPAINTING_METHOD=auto` for flexibility

**Total cost for entire project: ~$0.13 (3,000 VNĐ)**

Rất rẻ so với giá trị demo tốt!
