"""PDF processing utilities for the meta-review pipeline."""

import base64
import io
import os
from typing import Dict, List, Optional

import pymupdf
from PIL import Image


def pdf_to_images(pdf_path: str, max_pages: int = 9) -> List[Dict]:
    """Convert PDF pages to base64 encoded PNG images for multimodal prompts."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    images: List[Dict] = []
    doc = pymupdf.open(pdf_path)

    num_pages = min(len(doc), max_pages)
    print(f"Converting {num_pages} pages from {os.path.basename(pdf_path)}...")

    for page_num in range(num_pages):
        page = doc.load_page(page_num)

        mat = pymupdf.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)

        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))

        buffer = io.BytesIO()
        img.save(buffer, format="PNG", optimize=True)
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        images.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_base64}"
            },
        })

        print(f"  ✓ Converted page {page_num + 1}")

    doc.close()
    return images


def read_text_file(file_path: str) -> str:
    """Read a UTF-8 text file, returning a placeholder string on error."""
    if not os.path.exists(file_path):
        return f"[File not found: {file_path}]"

    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except Exception as exc:  # pragma: no cover - filesystem dependent
        return f"[Error reading file {file_path}: {exc}]"


def _infer_forum_id_from_pdf(pdf_filename: str) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(pdf_filename))[0]
    if "_" in stem:
        candidate = stem.split("_", 1)[0]
        return candidate or None
    return None


def _extract_forum_id_from_text(text: str) -> Optional[str]:
    for line in text.splitlines():
        if "Forum ID:" in line:
            _, _, remainder = line.partition("Forum ID:")
            candidate = remainder.strip()
            if candidate:
                return candidate
    return None


def extract_submission_data(
    submission_folder: str,
    *,
    convert_to_images: bool = True,
) -> Dict:
    """Extract metadata, reviews, meta-reviews, and images from a submission folder."""
    data: Dict[str, Dict] = {
        "submission_id": os.path.basename(submission_folder),
        "paper_pdf": None,
        "reviews": {},
        "meta_reviews": {},
        "paper_images": [],
        "submission_discussion": None,
        "forum_id": None,
    }

    if not os.path.exists(submission_folder):
        raise FileNotFoundError(f"Submission folder not found: {submission_folder}")

    files = os.listdir(submission_folder)

    pdf_files = [f for f in files if f.endswith(".pdf")]
    if pdf_files:
        pdf_path = os.path.join(submission_folder, pdf_files[0])
        data["paper_pdf"] = pdf_path
        data["forum_id"] = _infer_forum_id_from_pdf(pdf_files[0])
        print(f"Found paper PDF: {pdf_files[0]}")
        if convert_to_images:
            data["paper_images"] = pdf_to_images(pdf_path)

    review_files = [f for f in files if f.startswith("review_") and f.endswith(".txt")]
    for review_file in sorted(review_files):
        review_path = os.path.join(submission_folder, review_file)
        review_num = review_file.replace("review_", "").replace(".txt", "")
        data["reviews"][review_num] = read_text_file(review_path)
        print(f"Found review: {review_file}")
        if data.get("forum_id") is None:
            inferred = _extract_forum_id_from_text(data["reviews"][review_num])
            if inferred:
                data["forum_id"] = inferred

    meta_review_files = [f for f in files if f.startswith("meta_review_") and f.endswith(".txt")]
    for meta_review_file in sorted(meta_review_files):
        meta_review_path = os.path.join(submission_folder, meta_review_file)
        meta_review_num = meta_review_file.replace("meta_review_", "").replace(".txt", "")
        data["meta_reviews"][meta_review_num] = read_text_file(meta_review_path)
        print(f"Found meta-review: {meta_review_file}")
        if data.get("forum_id") is None:
            inferred = _extract_forum_id_from_text(data["meta_reviews"][meta_review_num])
            if inferred:
                data["forum_id"] = inferred

    discussion_path = os.path.join(submission_folder, "submission_discussion.txt")
    if os.path.exists(discussion_path):
        data["submission_discussion"] = read_text_file(discussion_path)
        print("Found submission_discussion.txt")

    return data
