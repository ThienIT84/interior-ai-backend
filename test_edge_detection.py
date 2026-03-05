"""
Test script cho edge_detection module (Task 3.1.2)
===================================================
Chay script nay tu thu muc backend de test:

    cd /home/tran_thien/interior_project/backend
    python test_edge_detection.py

Ket qua se duoc luu vao: data/outputs/test_edges_*.png
"""

import sys
import os
from pathlib import Path

# Them backend vao Python path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image

# Import cac ham can test
from app.core.edge_detection import (
    auto_canny_edges,
    detect_canny_edges,
    get_edge_density,
    validate_image_for_edges,
    edges_to_rgb,
)


def print_separator(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def find_test_image(backend_dir: Path) -> Path:
    """Tim anh test co san trong du an"""
    # Thu tim ảnh từ outputs (inpainted result) truoc
    for pattern in ["data/outputs/*.png", "data/outputs/*.jpg",
                    "data/inputs/*.jpg", "data/inputs/*.png"]:
        images = sorted((backend_dir / pattern.split("/")[0] /
                         pattern.split("/")[1]).glob(pattern.split("/")[-1]))
        if images:
            return images[0]
    return None


def test_validate():
    print_separator("Test 1: Image Validation")

    # Test voi anh hop le
    valid_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    ok, msg = validate_image_for_edges(valid_img)
    assert ok, f"Phai hop le: {msg}"
    print(f"  ✅ Valid image 640x480 RGB: OK")

    # Test voi anh qua nho
    tiny = np.zeros((32, 32, 3), dtype=np.uint8)
    ok, msg = validate_image_for_edges(tiny)
    assert not ok, "Phai fail voi anh nho"
    print(f"  ✅ Tiny image 32x32: Reject dung cach -> '{msg}'")

    # Test voi None
    ok, msg = validate_image_for_edges(None)
    assert not ok
    print(f"  ✅ None image: Reject dung cach -> '{msg}'")


def test_auto_canny(image_np: np.ndarray, output_dir: Path):
    print_separator("Test 2: Auto Canny Edge Detection")

    for sigma in [0.2, 0.33, 0.5]:
        edges = auto_canny_edges(image_np, sigma=sigma)
        density = get_edge_density(edges)
        status = "✅ IDEAL" if 5.0 <= density <= 15.0 else ("⚠️ LOW" if density < 5.0 else "⚠️ HIGH")
        print(f"  sigma={sigma:.2f}: density={density:.2f}% {status}")

        if sigma == 0.33:
            # Luu ket qua sigma default
            edge_rgb = edges_to_rgb(edges)
            out_path = output_dir / "test_edges_auto.png"
            Image.fromarray(edge_rgb).save(out_path)
            print(f"  💾 Saved: {out_path}")


def test_manual_canny(image_np: np.ndarray, output_dir: Path):
    print_separator("Test 3: Manual Canny Edge Detection")

    configs = [
        (30, 90,   "Soft edges"),
        (50, 150,  "Default (balanced)"),
        (100, 200, "Sharp edges"),
    ]

    for low, high, label in configs:
        edges = detect_canny_edges(image_np, low_threshold=low, high_threshold=high)
        density = get_edge_density(edges)
        status = "✅ IDEAL" if 5.0 <= density <= 15.0 else ("⚠️ LOW" if density < 5.0 else "⚠️ HIGH")
        print(f"  low={low}, high={high} ({label}): density={density:.2f}% {status}")

    # Luu ket qua config default
    edges = detect_canny_edges(image_np, low_threshold=50, high_threshold=150)
    edge_rgb = edges_to_rgb(edges)
    out_path = output_dir / "test_edges_manual.png"
    Image.fromarray(edge_rgb).save(out_path)
    print(f"  💾 Saved: {out_path}")


def test_with_synthetic_image(output_dir: Path):
    print_separator("Test 4: Synthetic Room Image (do khi khong co anh that)")

    # Tao anh gia lap phong don gian (nen trang, vien den)
    h, w = 480, 640
    img = np.ones((h, w, 3), dtype=np.uint8) * 200  # nen xam nhat

    # Ve tuong (vien)
    img[0:10, :] = 50       # tuong tren
    img[h-10:h, :] = 50     # san
    img[:, 0:10] = 50       # tuong trai
    img[:, w-10:w] = 50     # tuong phai

    # Ve cua so
    img[50:150, 200:350] = 240   # cua so sang
    img[50:150, 205:345] = 50    # vien cua so

    # Ve ghe (vat the)
    img[300:450, 400:550] = 80

    edges = auto_canny_edges(img)
    density = get_edge_density(edges)
    print(f"  Synthetic room image 640x480: density={density:.2f}%")

    edge_rgb = edges_to_rgb(edges)
    out_path = output_dir / "test_edges_synthetic.png"
    Image.fromarray(img).save(output_dir / "test_synthetic_input.png")
    Image.fromarray(edge_rgb).save(out_path)
    print(f"  💾 Saved synthetic input + edges")


def main():
    print("\n🔍 EDGE DETECTION MODULE TEST")
    print("Task 3.1.2 - Testing auto_canny_edges & detect_canny_edges")

    backend_dir = Path(__file__).parent
    output_dir = backend_dir / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test validation
    test_validate()

    # Tim anh that de test
    test_image_path = find_test_image(backend_dir)

    if test_image_path:
        print(f"\n📂 Using real image: {test_image_path.name}")
        image_pil = Image.open(test_image_path).convert("RGB")
        image_np = np.array(image_pil, dtype=np.uint8)
        print(f"   Size: {image_np.shape[1]}x{image_np.shape[0]}")

        test_auto_canny(image_np, output_dir)
        test_manual_canny(image_np, output_dir)
    else:
        print("\n⚠️  Khong tim thay anh that, dung anh tong hop")
        test_with_synthetic_image(output_dir)

    # Luon test voi anh tong hop
    test_with_synthetic_image(output_dir)

    print("\n" + "=" * 60)
    print("  ✅ TAT CA TESTS HOAN THANH")
    print(f"  📁 Ket qua o: {output_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
