"""
Prompt Library for AI Interior Design
======================================
Tập trung tất cả prompts cho Inpainting và ControlNet Generation.

Structure:
    INPAINTING_PROMPTS  - Dùng cho object removal (Week 2)
    STYLE_PROMPTS       - Dùng cho ControlNet design generation (Week 3)
    AVAILABLE_STYLES    - Danh sách style hợp lệ cho validation

Usage:
    from app.core.prompts import get_style_prompts, INPAINTING_PROMPTS, AVAILABLE_STYLES

    # ControlNet generation
    pos, neg = get_style_prompts("modern")

    # Inpainting
    pos = INPAINTING_PROMPTS["positive"]
    neg = INPAINTING_PROMPTS["negative"]
"""

from typing import Tuple

# ---------------------------------------------------------------------------
# Inpainting Prompts (Task 2.1 - Object Removal)
# ---------------------------------------------------------------------------

INPAINTING_PROMPTS = {
    "positive": (
        "empty room, bare floor, plain wall, no furniture, no objects, "
        "clean surface, seamless background, matching floor texture, "
        "continuous wall pattern, natural lighting, photorealistic, 8k"
    ),
    "negative": (
        "furniture, chair, table, sofa, bed, desk, shelf, cabinet, lamp, "
        "new furniture, replaced furniture, upgraded furniture, different chair, "
        "object, decoration, person, people, animal, pet, "
        "leftover shadows, floating objects, ghosting, mismatched patterns, "
        "harsh edges, cut seams, text, watermark, "
        "blurry, deformed, cartoon, illustration, low quality"
    ),
}

# ---------------------------------------------------------------------------
# Style Prompts for ControlNet Generation (Task 3.2.3)
# ---------------------------------------------------------------------------

_STYLE_CONFIGS = {
    "modern": {
        "display_name": "Modern",
        "description": "Phong cách hiện đại với đường nét sạch, nội thất tối giản và ánh sáng tự nhiên",
        "positive": (
            "modern interior design, minimalist furniture, clean lines, neutral colors, "
            "natural light, scandinavian style, open space, white walls, wooden floor, "
            "contemporary decor, high quality, photorealistic, architectural photography, 8k"
        ),
        "negative": (
            "cluttered, dark, ornate, victorian, vintage, antique, rustic, traditional, "
            "old fashioned, heavy furniture, dark wood, busy patterns, "
            "person, people, text, watermark, blurry, low quality, cartoon, illustration"
        ),
        "additional_positive": "best quality, extremely detailed, sharp focus, professional lighting",
    },
    "minimalist": {
        "display_name": "Minimalist",
        "description": "Phong cách tối giản thuần túy, không gian mở, vật liệu tự nhiên",
        "positive": (
            "minimalist interior, white walls, simple furniture, open space, natural materials, "
            "zen, peaceful, japandi style, neutral tones, linen, wood, stone, "
            "airy, decluttered, functional, clean aesthetic, photorealistic, 8k"
        ),
        "negative": (
            "cluttered, decorated, colorful, ornate, maximalist, busy, "
            "too many objects, loud colors, heavy drapes, pattern wallpaper, "
            "baroque, rococo, eclectic, bohemian, "
            "person, people, text, watermark, blurry, low quality"
        ),
        "additional_positive": "best quality, extremely detailed, serene atmosphere, soft natural light",
    },
    "industrial": {
        "display_name": "Industrial",
        "description": "Phong cách công nghiệp với gạch trần, kim loại, sàn bê tông",
        "positive": (
            "industrial interior, exposed brick wall, metal furniture, concrete floor, "
            "loft style, urban, raw materials, steel beams, Edison bulbs, "
            "dark tones, warehouse aesthetic, unfinished look, masculine, "
            "photorealistic, architectural photography, 8k"
        ),
        "negative": (
            "soft, feminine, pastel, floral, traditional, ornate, white walls, "
            "polished marble, luxury, glamorous, boho, rustic farmhouse, "
            "too bright, colorful, decorated, "
            "person, people, text, watermark, blurry, low quality"
        ),
        "additional_positive": "best quality, extremely detailed, dramatic shadows, moody lighting",
    },
}

# Public: list of valid style names
AVAILABLE_STYLES = list(_STYLE_CONFIGS.keys())


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_style_prompts(style: str) -> Tuple[str, str]:
    """
    Lấy (positive_prompt, negative_prompt) cho một style.

    Args:
        style: Một trong AVAILABLE_STYLES ("modern", "minimalist", "industrial")

    Returns:
        Tuple (positive_prompt, negative_prompt)

    Raises:
        ValueError: Nếu style không hợp lệ
    """
    style = style.lower().strip()
    if style not in _STYLE_CONFIGS:
        raise ValueError(
            f"Style '{style}' không hợp lệ. "
            f"Các style hỗ trợ: {AVAILABLE_STYLES}"
        )
    cfg = _STYLE_CONFIGS[style]
    positive = cfg["positive"] + ", " + cfg["additional_positive"]
    negative = cfg["negative"]
    return positive, negative


def get_style_info(style: str) -> dict:
    """
    Lấy toàn bộ thông tin (tên hiển thị, mô tả, prompts) của một style.

    Args:
        style: Tên style

    Returns:
        Dict với keys: display_name, description, positive, negative

    Raises:
        ValueError: Nếu style không hợp lệ
    """
    style = style.lower().strip()
    if style not in _STYLE_CONFIGS:
        raise ValueError(f"Style '{style}' không hợp lệ. Các style hỗ trợ: {AVAILABLE_STYLES}")
    cfg = _STYLE_CONFIGS[style]
    pos, neg = get_style_prompts(style)
    return {
        "name": style,
        "display_name": cfg["display_name"],
        "description": cfg["description"],
        "positive_prompt": pos,
        "negative_prompt": neg,
    }


def list_styles() -> list[dict]:
    """
    Lấy danh sách tất cả styles với thông tin hiển thị.

    Returns:
        List[dict] mỗi dict có: name, display_name, description
    """
    return [
        {
            "name": key,
            "display_name": cfg["display_name"],
            "description": cfg["description"],
        }
        for key, cfg in _STYLE_CONFIGS.items()
    ]
