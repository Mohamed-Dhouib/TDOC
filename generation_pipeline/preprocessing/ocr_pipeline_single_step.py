import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from google.protobuf.json_format import MessageToDict
from tqdm import tqdm

try:  # Optional dependency for Google Vision
    from google.cloud import vision  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    vision = None  # type: ignore

try:  # Optional dependency for Tesseract
    import pytesseract  # type: ignore
    from pytesseract import Output as PytesseractOutput  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    pytesseract = None  # type: ignore
    PytesseractOutput = None  # type: ignore

POSSIBLE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}


@lru_cache(maxsize=1)
def _get_client() -> Any:
    if vision is None:
        raise RuntimeError(
            "Google Cloud Vision is not available. Install 'google-cloud-vision' and set credentials, or use the tesseract backend."
        )
    return vision.ImageAnnotatorClient()


def _ensure_tesseract_available() -> None:
    if pytesseract is None:
        raise RuntimeError(
            "Tesseract backend requested but 'pytesseract' is not installed. Install it or switch to the google backend."
        )


def _format_bounding_box(bounding_box: Dict) -> Optional[Dict[str, int]]:
    vertices = bounding_box.get("vertices", [])
    if len(vertices) < 4:
        return None

    xs = [int(v.get("x", 0)) for v in vertices]
    ys = [int(v.get("y", 0)) for v in vertices]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    if x_min == x_max or y_min == y_max:
        return None

    return {
        "W_d": x_min,
        "W_g": x_max,
        "H_b": y_min,
        "H_h": y_max,
    }


def _extract_words_and_characters(annotation: Dict) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    words: List[Dict[str, Any]] = []
    characters: List[Dict[str, Any]] = []
    if not annotation:
        return words, characters

    for page in annotation.get("pages", []):
        for block in page.get("blocks", []):
            for paragraph in block.get("paragraphs", []):
                for word in paragraph.get("words", []):
                    symbols = word.get("symbols", [])
                    text = "".join(symbol.get("text", "") for symbol in symbols)
                    bounding_box = _format_bounding_box(word.get("boundingBox", {}))

                    if text and bounding_box:
                        words.append({
                            "Word": text,
                            "BoundingBox": bounding_box,
                        })

                    for symbol in symbols:
                        char_text = symbol.get("text", "")
                        char_box = _format_bounding_box(symbol.get("boundingBox", {}))
                        if char_text and char_box:
                            characters.append({
                                "Word": char_text,
                                "BoundingBox": char_box,
                            })
    return words, characters


def _intersection_area(b1: Dict[str, int], b2: Dict[str, int]) -> int:
    left = max(b1["W_d"], b2["W_d"])
    right = min(b1["W_g"], b2["W_g"])
    top = max(b1["H_b"], b2["H_b"])
    bottom = min(b1["H_h"], b2["H_h"])

    if right <= left or bottom <= top:
        return 0
    return (right - left) * (bottom - top)


def _deduplicate_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []

    for i, item in enumerate(entries):
        bbox = item["BoundingBox"]
        width = bbox["W_g"] - bbox["W_d"]
        height = bbox["H_h"] - bbox["H_b"]
        area1 = width * height
        if area1 <= 0:
            continue

        keep = True
        for j, other in enumerate(entries):
            if i == j:
                continue
            other_bbox = other["BoundingBox"]
            o_width = other_bbox["W_g"] - other_bbox["W_d"]
            o_height = other_bbox["H_h"] - other_bbox["H_b"]
            area2 = o_width * o_height
            if area2 <= 0:
                continue

            intersect = _intersection_area(bbox, other_bbox)
            if intersect >= 0.5 * area1 and intersect >= 0.5 * area2:
                keep = False
                break
        if keep:
            filtered.append(item)

    return filtered


def _expand_boxes(entries: List[Dict[str, Any]], image_size: Tuple[int, int], padding: int) -> None:
    height, width = image_size
    for item in entries:
        bbox = item["BoundingBox"]
        bbox["W_d"] = max(0, bbox["W_d"] - padding)
        bbox["W_g"] = min(width, bbox["W_g"] + padding)
        bbox["H_b"] = max(0, bbox["H_b"] - padding)
        bbox["H_h"] = min(height, bbox["H_h"] + padding)


def _bbox_signature(entry: Dict[str, Any]) -> Tuple[str, Tuple[int, int, int, int]]:
    word = str(entry.get("Word", ""))
    bbox = entry["BoundingBox"]
    return (
        word,
        (
            bbox["W_d"],
            bbox["H_b"],
            bbox["W_g"],
            bbox["H_h"],
        ),
    )


def _remove_word_character_duplicates(
    words: List[Dict[str, Any]],
    characters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    word_signatures = {_bbox_signature(word) for word in words}
    seen_signatures = set(word_signatures)
    filtered_chars: List[Dict[str, Dict[str, int]]] = []

    for char in characters:
        signature = _bbox_signature(char)
        # Skip characters whose signature already appears in words to avoid duplicates
        if signature in word_signatures:
            continue
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        filtered_chars.append(char)

    return filtered_chars


def _run_document_text_detection(image_bytes: bytes) -> Dict:
    if vision is None:
        raise RuntimeError(
            "Google Cloud Vision is not available. Install 'google-cloud-vision' and set credentials, or use the tesseract backend."
        )
    client = _get_client()
    image = vision.Image(content=image_bytes)
    response = client.document_text_detection(image=image)
    if response.error.message:
        raise RuntimeError(response.error.message)
    response_dict = MessageToDict(response._pb)
    return response_dict.get("fullTextAnnotation", {})


def _run_tesseract_ocr(image, lang: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    _ensure_tesseract_available()
    if pytesseract is None or PytesseractOutput is None:
        raise RuntimeError("pytesseract Output helper unavailable. Update pytesseract installation.")

    tess = pytesseract

    words: List[Dict[str, Any]] = []
    data = tess.image_to_data(image, lang=lang, output_type=PytesseractOutput.DICT)
    if data and "text" in data:
        n_items = len(data["text"])
        for i in range(n_items):
            text = data["text"][i].strip()
            if not text:
                continue
            width = int(data["width"][i])
            height = int(data["height"][i])
            if width <= 0 or height <= 0:
                continue
            left = int(data["left"][i])
            top = int(data["top"][i])
            words.append(
                {
                    "Word": text,
                    "BoundingBox": {
                        "W_d": max(0, left),
                        "W_g": max(0, left + width),
                        "H_b": max(0, top),
                        "H_h": max(0, top + height),
                    },
                }
            )

    characters: List[Dict[str, Any]] = []
    boxes_raw = tess.image_to_boxes(image, lang=lang)
    if boxes_raw:
        img_height = int(image.shape[0])
        for entry in boxes_raw.strip().splitlines():
            parts = entry.split()
            if len(parts) < 5:
                continue
            char = parts[0]
            if not char.strip():
                continue
            try:
                x1, y1, x2, y2 = map(int, parts[1:5])
            except ValueError:
                continue
            # Tesseract coordinates start at bottom-left
            top = img_height - y2
            bottom = img_height - y1
            characters.append(
                {
                    "Word": char,
                    "BoundingBox": {
                        "W_d": max(0, x1),
                        "W_g": max(0, x2),
                        "H_b": max(0, top),
                        "H_h": max(0, bottom),
                    },
                }
            )

    return words, characters


def process_image(
    image_path: Path,
    output_dir: Path,
    padding: int,
    skip_existing: bool,
    backend: str,
    tesseract_lang: str,
) -> Tuple[str, bool, str]:
    output_path = output_dir / f"{image_path.stem}.json"
    if skip_existing and output_path.exists():
        return (image_path.name, True, "skipped (existing)")

    image = cv2.imread(str(image_path))
    if image is None:
        return (image_path.name, False, "failed to read image")

    try:
        if backend == "google":
            with image_path.open("rb") as f:
                annotation = _run_document_text_detection(f.read())
            words, characters = _extract_words_and_characters(annotation)
        else:
            words, characters = _run_tesseract_ocr(image, lang=tesseract_lang)
    except Exception as exc:  # pylint: disable=broad-except
        return (image_path.name, False, f"{backend} error: {exc}")

    words = _deduplicate_entries(words)
    characters = _deduplicate_entries(characters)
    characters = _remove_word_character_duplicates(words, characters)
    if padding:
        _expand_boxes(words, image.shape[:2], padding)
        _expand_boxes(characters, image.shape[:2], padding)

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_entries = words + characters
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged_entries, f, ensure_ascii=False, indent=2)

    return (image_path.name, True, f"saved {len(merged_entries)} boxes")


def gather_images(image_dir: Path) -> List[Path]:
    return [
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower().lstrip(".") in POSSIBLE_EXTENSIONS
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-step OCR pipeline producing merged JSON outputs.")
    parser.add_argument("--image-dir", default="images", type=Path, help="Directory containing input images.")
    parser.add_argument("--output-dir", default="merged_jsons", type=Path, help="Directory to store final merged JSON files.")
    parser.add_argument("--padding", type=int, default=1, help="Pixels to expand each bounding box on every side.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers to use.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip processing images whose outputs already exist.")
    parser.add_argument(
        "--ocr-backend",
        choices=["google", "tesseract"],
        default="google",
        help="OCR provider to use. Default: google",
    )
    parser.add_argument(
        "--tesseract-lang",
        default="eng",
        help="Language(s) to pass to Tesseract when using the tesseract backend.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_dir: Path = args.image_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = args.ocr_backend
    if backend == "tesseract":
        _ensure_tesseract_available()

    image_paths = gather_images(image_dir)
    if not image_paths:
        print(f"No images found in {image_dir} with extensions: {', '.join(sorted(POSSIBLE_EXTENSIONS))}")
        return

    worker = partial(
        process_image,
        output_dir=output_dir,
        padding=max(0, args.padding),
        skip_existing=args.skip_existing,
        backend=backend,
        tesseract_lang=args.tesseract_lang,
    )

    num_workers = max(1, args.num_workers)

    results: List[Tuple[str, bool, str]] = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(worker, path): path for path in image_paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            try:
                results.append(future.result())
            except Exception as exc:  # pylint: disable=broad-except
                results.append((str(futures[future].name), False, f"unexpected error: {exc}"))

    successes = sum(1 for _, ok, _ in results if ok)
    failures = [msg for _, ok, msg in results if not ok]

    print(f"Completed {successes}/{len(results)} images.")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure}")


if __name__ == "__main__":
    main()
