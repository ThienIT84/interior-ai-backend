"""
Mask Quality Gate + Preprocessing Utilities
============================================
Dùng trước LaMa để đảm bảo mask hợp lệ, và sau LaMa để phát hiện artifact.

Flow chuẩn:
    1. validate_mask_coverage()       -- reject mask quá nhỏ/quá lớn
    2. preprocess_mask_for_removal()  -- morphology close/open + feather
    3. (sau LaMa) compute_artifact_score() -- đo seam/mismatch quanh biên
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter
from typing import Tuple, Optional
from dataclasses import dataclass

from app.utils.logger import logger


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MaskValidationResult:
    valid: bool
    coverage_pct: float
    reason: Optional[str] = None   # mô tả lỗi nếu valid=False


@dataclass
class MaskPreprocessResult:
    mask: Image.Image               # mask đã xử lý
    original_coverage_pct: float
    processed_coverage_pct: float
    dilation_px: int
    feather_px: int


# ---------------------------------------------------------------------------
# 1. Coverage Gate
# ---------------------------------------------------------------------------

def validate_mask_coverage(
    mask: Image.Image,
    min_fraction: float = 0.003,
    max_fraction: float = 0.45,
) -> MaskValidationResult:
    """
    Kiểm tra xem mask có hợp lệ để gửi LaMa không.

    Args:
        mask:         PIL Image grayscale (L) hoặc có thể convert được.
        min_fraction: Tỉ lệ tối thiểu ảnh được che (ví dụ 0.003 = 0.3%).
        max_fraction: Tỉ lệ tối đa cho phép (ví dụ 0.45 = 45%).

    Returns:
        MaskValidationResult
    """
    mask_np = np.array(mask.convert("L"))
    total_pixels = mask_np.size
    white_pixels = int((mask_np > 127).sum())
    coverage = white_pixels / total_pixels

    if coverage < min_fraction:
        return MaskValidationResult(
            valid=False,
            coverage_pct=coverage * 100,
            reason=(
                f"Mask quá nhỏ ({coverage*100:.2f}% < {min_fraction*100:.1f}%). "
                "Hãy chạm thêm điểm hoặc mô tả rõ hơn vật thể."
            ),
        )

    if coverage > max_fraction:
        return MaskValidationResult(
            valid=False,
            coverage_pct=coverage * 100,
            reason=(
                f"Mask quá lớn ({coverage*100:.2f}% > {max_fraction*100:.1f}%). "
                "Hãy chọn vùng nhỏ hơn để kết quả tốt hơn."
            ),
        )

    logger.info(f"✅ Mask coverage OK: {coverage*100:.2f}%")
    return MaskValidationResult(valid=True, coverage_pct=coverage * 100)


# ---------------------------------------------------------------------------
# 2. Preprocess (morphology + feather)
# ---------------------------------------------------------------------------

def preprocess_mask_for_removal(
    mask: Image.Image,
    dilation_px: int = 15,
    feather_px: int = 4,
) -> MaskPreprocessResult:
    """
    Chuẩn hóa mask trước khi gửi LaMa:
      Step 1 - Binarize
      Step 2 - Morphology close  (lấp lỗ nhỏ bên trong)
      Step 3 - Morphology open   (loại bỏ noise nhỏ bên ngoài)
      Step 4 - Dilate             (mở rộng để ăn viền + bóng đổ)
      Step 5 - Feather            (làm mềm biên để hòa màu tốt hơn)

    Args:
        mask:        PIL Image (bất kỳ mode nào, sẽ convert sang L).
        dilation_px: Số pixel mở rộng mask.
        feather_px:  Bán kính feather (làm mềm biên).

    Returns:
        MaskPreprocessResult
    """
    # --- Binarize ---
    mask_np = np.array(mask.convert("L"))
    original_coverage = float((mask_np > 127).sum()) / mask_np.size
    binary = (mask_np > 127).astype(np.uint8) * 255

    # --- Close: lấp lỗ nhỏ bên trong object ---
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    # --- Open: loại bỏ noise pixel đơn lẻ ---
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)

    # --- Dilate ---
    if dilation_px > 0:
        dilate_kernel_size = dilation_px * 2 + 1
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size)
        )
        dilated = cv2.dilate(opened, dilate_kernel, iterations=1)
    else:
        dilated = opened

    # --- Feather (blur biên) ---
    if feather_px > 0:
        blur_size = feather_px * 2 + 1
        feathered = cv2.GaussianBlur(dilated.astype(np.float32), (blur_size, blur_size), feather_px)
        # Re-threshold để không tạo gradient quá nhạt ở phần trong
        feathered = np.clip(feathered, 0, 255).astype(np.uint8)
    else:
        feathered = dilated

    processed_coverage = float((feathered > 127).sum()) / feathered.size
    result_mask = Image.fromarray(feathered, mode="L")

    logger.info(
        f"🎭 Mask preprocessed: {original_coverage*100:.1f}% → {processed_coverage*100:.1f}% "
        f"(dilation={dilation_px}px, feather={feather_px}px)"
    )

    return MaskPreprocessResult(
        mask=result_mask,
        original_coverage_pct=original_coverage * 100,
        processed_coverage_pct=processed_coverage * 100,
        dilation_px=dilation_px,
        feather_px=feather_px,
    )


# ---------------------------------------------------------------------------
# 3. Artifact Score (post-LaMa quality check)
# ---------------------------------------------------------------------------

def compute_artifact_score(
    original: Image.Image,
    result: Image.Image,
    mask: Image.Image,
    ring_px: int = 20,
) -> float:
    """
    Đo mức độ artifact (seam/mismatch) tại vùng biên của mask sau LaMa.

    Thuật toán:
      - Tạo "ring" quanh viền mask (dilation - erosion).
      - Tính gradient magnitude sigma (std deviation) tại ring của ảnh gốc.
      - Tính gradient magnitude sigma tại ring của ảnh kết quả.
      - Score = max(0, (sigma_result - sigma_original) / (sigma_original + 1e-6))
      - Range: 0 (hoàn hảo) đến ~1+ (nhiều artifact)

    Args:
        original:  Ảnh gốc RGB (PIL).
        result:    Ảnh sau LaMa RGB (PIL).
        mask:      Mask đã dùng (grayscale PIL).
        ring_px:   Độ rộng ring kiểm tra quanh biên (pixels).

    Returns:
        score (float, 0 = tốt, > threshold = cần repair)
    """
    # Normalize sizes
    if result.size != original.size:
        result = result.resize(original.size, Image.LANCZOS)
    if mask.size != original.size:
        mask = mask.resize(original.size, Image.NEAREST)

    mask_np = (np.array(mask.convert("L")) > 127).astype(np.uint8)

    # Build ring: dilated XOR eroded
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_px * 2 + 1, ring_px * 2 + 1))
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(mask_np, ring_kernel)
    eroded = cv2.erode(mask_np, small_kernel)
    ring = (dilated - eroded).astype(bool)

    if ring.sum() < 100:
        # Ring quá nhỏ, không đủ sample để đo
        return 0.0

    # Convert to grayscale numpy
    orig_gray = np.array(original.convert("L")).astype(np.float32)
    res_gray = np.array(result.convert("L")).astype(np.float32)

    # Laplacian (gradient magnitude proxy)
    orig_lap = np.abs(cv2.Laplacian(orig_gray, cv2.CV_32F))
    res_lap = np.abs(cv2.Laplacian(res_gray, cv2.CV_32F))

    sigma_orig = float(orig_lap[ring].std())
    sigma_res = float(res_lap[ring].std())

    score = max(0.0, (sigma_res - sigma_orig) / (sigma_orig + 1e-6))

    logger.info(
        f"🔍 Artifact score: {score:.3f} "
        f"(ring_orig_σ={sigma_orig:.2f}, ring_result_σ={sigma_res:.2f})"
    )
    return score


# ---------------------------------------------------------------------------
# 4. Build artifact repair mask
# ---------------------------------------------------------------------------

def build_artifact_repair_mask(
    original: Image.Image,
    result: Image.Image,
    primary_mask: Image.Image,
    ring_px: int = 20,
    threshold_pct: float = 20.0,
) -> Optional[Image.Image]:
    """
    Tạo mask nhỏ chỉ vùng bị artifact để chạy SD inpainting pass 2.

    Returns:
        PIL Image grayscale hoặc None nếu không phát hiện artifact đáng kể.
    """
    if result.size != original.size:
        result = result.resize(original.size, Image.LANCZOS)

    orig_gray = np.array(original.convert("L")).astype(np.float32)
    res_gray = np.array(result.convert("L")).astype(np.float32)

    # Pixel-level absolute difference
    diff = np.abs(orig_gray - res_gray)

    # Tập trung vào ring quanh mask
    mask_np = (np.array(primary_mask.convert("L")) > 127).astype(np.uint8)
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_px * 2 + 1, ring_px * 2 + 1))
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(mask_np, ring_kernel)
    eroded = cv2.erode(mask_np, small_kernel)
    ring = (dilated - eroded).astype(bool)

    if ring.sum() == 0:
        return None

    # Threshold: pixel difference > percentile được coi là artifact
    ring_diffs = diff[ring]
    threshold = float(np.percentile(ring_diffs, threshold_pct))
    artifact_px = np.zeros_like(mask_np, dtype=np.uint8)
    artifact_px[ring & (diff > threshold)] = 255

    # Dilate nhỏ để phủ đủ vùng sửa
    pad_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    artifact_px = cv2.dilate(artifact_px, pad_kernel)

    if artifact_px.sum() < 500:   # quá ít pixel, bỏ qua
        return None

    logger.info(f"🔧 Artifact repair mask built: {artifact_px.sum()//255} px")
    return Image.fromarray(artifact_px, mode="L")
