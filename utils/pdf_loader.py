import os
import fitz  # PyMuPDF

def _extract_page_images(page: fitz.Page) -> list[dict]:

    images = []
    doc = page.parent  # get the parent Document object

    seen_xrefs = set()  # avoid duplicates (same image referenced twice on page)

    for img_index, img_info in enumerate(page.get_images(full=True)):
        xref = img_info[0]  # first element is always the xref

        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)

        try:
            image_data = doc.extract_image(xref)
        except Exception:
            continue  # skip unreadable/corrupt images silently

        width  = image_data["width"]
        height = image_data["height"]

        # Skip tiny decorative images (icons, bullets, watermarks)
        if width < 100 or height < 100:
            continue

        images.append({
            "image_index": img_index,
            "ext":    image_data["ext"],   
            "width":  width,
            "height": height,
            "data":   image_data["image"]   # raw bytes
        })

    return images


def _save_images(images: list[dict], output_dir: str, prefix: str) -> list[str]:

    os.makedirs(output_dir, exist_ok=True)
    paths = []

    for img in images:
        filename = f"{prefix}_img{img['image_index']}.{img['ext']}"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "wb") as f:
            f.write(img["data"])

        paths.append(filepath)

    return paths


# ── Public API ────────────────────────────────────────────────────────────────

def load_pdf(filepath: str, extract_images: bool = True) -> list[dict]:

    pages = []
    source_name = os.path.basename(filepath)
    doc = fitz.open(filepath)

    for page_index in range(len(doc)):
        page = doc[page_index]

        text   = page.get_text("text").strip()
        images = _extract_page_images(page) if extract_images else []

        # Skip pages with no text AND no images
        if not text and not images:
            continue

        pages.append({
            "page_number": page_index + 1,
            "text":        text,
            "source":      source_name,
            "images":      images
        })

    doc.close()
    return pages


def load_pdf_with_metadata(
    filepath: str,
    extract_images: bool = True
) -> tuple[list[dict], dict]:
   
    source_name = os.path.basename(filepath)
    doc = fitz.open(filepath)

    metadata = {
        "title":      doc.metadata.get("title", "Unknown"),
        "author":     doc.metadata.get("author", "Unknown"),
        "page_count": len(doc),
        "source":     source_name
    }

    pages = []
    for page_index in range(len(doc)):
        page = doc[page_index]

        text   = page.get_text("text").strip()
        images = _extract_page_images(page) if extract_images else []

        if not text and not images:
            continue

        pages.append({
            "page_number": page_index + 1,
            "text":        text,
            "source":      source_name,
            "images":      images
        })

    doc.close()
    return pages, metadata


def save_all_images(pages: list[dict], output_dir: str = "extracted_images") -> dict:
 
    saved = {}

    for page in pages:
        if not page["images"]:
            continue

        prefix = f"{page['source'].replace('.pdf', '')}_p{page['page_number']}"
        paths  = _save_images(page["images"], output_dir, prefix)
        saved[page["page_number"]] = paths

    return saved

