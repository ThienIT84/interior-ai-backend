"""
Prompt Library for AI Interior Design
======================================
Tập trung tất cả prompts cho Inpainting và ControlNet Generation.

Structure:
    INPAINTING_PROMPTS  - Dùng cho object removal (Week 2)
    STYLE_PROMPTS       - Dùng cho ControlNet design generation (Week 3)
    AVAILABLE_STYLES    - Danh sách style hợp lệ cho validation
    FURNITURE_PLACEMENT_NEGATIVE  - Negative prompt cho Option 1 (furniture placement)

Usage:
    from app.core.prompts import get_style_prompts, INPAINTING_PROMPTS, AVAILABLE_STYLES
    from app.core.prompts import get_furniture_placement_prompt

    # ControlNet generation
    pos, neg = get_style_prompts("modern")

    # Furniture placement
    pos, neg = get_furniture_placement_prompt("a modern leather sofa")
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
    "indochine": {
        "display_name": "Indochine",
        "description": "Phong cách Đông Dương: gỗ nhiệt đới, họa tiết hoa văn, ánh vàng ấm áp",
        "positive": (
            "indochine interior design, vietnamese colonial style, tropical wood furniture, "
            "rattan and bamboo accents, warm golden lighting, terracotta floor tiles, "
            "silk curtains, hand-carved wooden panels, lush indoor plants, "
            "deep warm tones, rich textures, copper and brass details, "
            "french colonial architecture, high ceiling, ceiling fan, "
            "photorealistic, architectural photography, 8k"
        ),
        "negative": (
            "modern, minimalist, scandinavian, industrial, cold colors, "
            "glass and steel, white walls, concrete, plastic furniture, "
            "person, people, text, watermark, blurry, low quality, cartoon"
        ),
        "additional_positive": "best quality, extremely detailed, warm atmosphere, cinematic lighting",
    },
    "scandinavian": {
        "display_name": "Scandinavian",
        "description": "Phong cách Bắc Âu: sáng, ấm, chức năng, hòa hợp với thiên nhiên",
        "positive": (
            "scandinavian interior design, hygge aesthetic, light oak wood floor, "
            "white and cream walls, cozy textiles, wool throws, sheepskin rugs, "
            "simple functional furniture, large windows, natural daylight, "
            "potted plants, subtle warm tones, birch wood, candles, "
            "nordic minimalism, clean lines, uncluttered, "
            "photorealistic, architectural photography, 8k"
        ),
        "negative": (
            "dark, heavy, ornate, baroque, industrial, tropical, "
            "bright paint colors, bold patterns, too many objects, "
            "concrete floors, exposed brick, metal furniture, "
            "person, people, text, watermark, blurry, low quality"
        ),
        "additional_positive": "best quality, extremely detailed, bright airy atmosphere, soft diffused light",
    },
}

# Public: list of valid style names
AVAILABLE_STYLES = list(_STYLE_CONFIGS.keys())

# ---------------------------------------------------------------------------
# Furniture Placement Prompts (Option 1 - Targeted Object Placement)
# ---------------------------------------------------------------------------

FURNITURE_PLACEMENT_NEGATIVE = (
    "distorted, deformed, unrealistic proportions, floating, shadow mismatch, "
    "wrong perspective, blurry, low quality, cartoon, illustration, painting, "
    "person, people, text, watermark, duplicate objects"
)


def get_furniture_placement_prompt(furniture_description: str) -> Tuple[str, str]:
    """
    Tạo prompt cho việc đặt đồ nội thất vào vị trí chọn.

    Args:
        furniture_description: Mô tả đồ nội thất (VD: "a modern leather sofa")

    Returns:
        Tuple (positive_prompt, negative_prompt)
    """
    furniture_description = furniture_description.strip()
    positive = (
        f"{furniture_description}, "
        "photorealistic, natural lighting, matching room style, "
        "correct perspective, proper shadows, high quality, 8k, "
        "seamlessly integrated, interior design photography"
    )
    return positive, FURNITURE_PLACEMENT_NEGATIVE


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
