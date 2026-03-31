from dotenv import load_dotenv
load_dotenv()
import io
import os
import pickle

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor


# ── Model loading ─────────────────────────────────────────────────────────────

# Lazy globals — models load once, stay in memory
_text_model  = None
_clip_model  = None
_clip_proc   = None


def _get_text_model() -> SentenceTransformer:
    global _text_model
    if _text_model is None:
        print("Loading text model (all-MiniLM-L6-v2)...")
        _text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _text_model


def _get_clip() -> tuple:
    global _clip_model, _clip_proc
    if _clip_model is None:
        print("Loading CLIP model (clip-vit-base-patch32)...")
        _clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
    return _clip_model, _clip_proc


# ── Embedding functions ───────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> np.ndarray:
    model = _get_text_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=len(texts) > 20,
        normalize_embeddings=True  # cosine similarity = dot product
    )
    return embeddings  # already np.ndarray


def embed_texts_clip(texts: list[str]) -> np.ndarray:

    import torch

    model, processor = _get_clip()

    all_embeddings = []
    batch_size = 32

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77  # CLIP's token limit
        )
        with torch.no_grad():
            features = model.get_text_features(**inputs)
            # get_text_features() returns a plain tensor in most versions,
            # but some transformers builds wrap it in a ModelOutput object.
            # Unwrap if needed, then L2-normalise.
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            features = torch.nn.functional.normalize(features, dim=-1)
        all_embeddings.append(features.cpu().numpy())

    return np.vstack(all_embeddings)


def embed_images_clip(image_bytes_list: list[bytes]) -> np.ndarray:

    import torch

    model, processor = _get_clip()

    pil_images = []
    for raw_bytes in image_bytes_list:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        pil_images.append(img)

    all_embeddings = []
    batch_size = 16  # images are heavier than text

    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt")

        with torch.no_grad():
            features = model.get_image_features(**inputs)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            features = torch.nn.functional.normalize(features, dim=-1)
        all_embeddings.append(features.cpu().numpy())

    return np.vstack(all_embeddings)


# ── Building the index ────────────────────────────────────────────────────────

def build_index(
    chunks: list[dict],
    pages:  list[dict],
    use_clip_for_text: bool = False
) -> dict:

    all_embeddings = []
    all_metadata   = []

    # ── 1. Embed text chunks ──────────────────────────────────────────────────
    print(f"Embedding {len(chunks)} text chunks...")
    texts = [c["text"] for c in chunks]

    if use_clip_for_text:
        text_embeddings = embed_texts_clip(texts)
    else:
        text_embeddings = embed_texts(texts)

    all_embeddings.append(text_embeddings)

    for chunk in chunks:
        all_metadata.append({
            "type":        "text",
            "content":     chunk["text"],
            "chunk_id":    chunk["chunk_id"],
            "page_number": chunk["page_number"],
            "source":      chunk["source"]
        })

    # ── 2. Embed images ───────────────────────────────────────────────────────
    image_bytes = []
    image_meta  = []

    for page in pages:
        for img in page.get("images", []):
            image_bytes.append(img["data"])
            image_meta.append({
                "type":        "image",
                "content":     f"image_p{page['page_number']}_i{img['image_index']}",
                "chunk_id":    None,
                "page_number": page["page_number"],
                "source":      page["source"],
                "ext":         img["ext"],
                "width":       img["width"],
                "height":      img["height"],
                "data":        img["data"]  # keep bytes for display later
            })

    if image_bytes:
        print(f"Embedding {len(image_bytes)} images with CLIP...")
        img_embeddings = embed_images_clip(image_bytes)

        # If text used sentence-transformers (384-dim), we can't directly mix
        # with CLIP image embeddings (512-dim). Store them separately in that case.
        if not use_clip_for_text and text_embeddings.shape[1] != img_embeddings.shape[1]:
            print(
                "Note: text embeddings are 384-dim (MiniLM) and image embeddings "
                "are 512-dim (CLIP). They live in different spaces. "
                "Pass use_clip_for_text=True to unify them."
            )
            # Still store them — retrieval functions handle each type separately
        all_embeddings.append(img_embeddings)
        all_metadata.extend(image_meta)
    else:
        print("No images found in pages — skipping image embedding.")

    # ── 3. Stack everything ───────────────────────────────────────────────────
    final_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])

    index = {
        "embeddings": final_embeddings,
        "metadata":   all_metadata,
        "use_clip_for_text": use_clip_for_text
    }

    print(f"Index built: {final_embeddings.shape[0]} total entries.")
    return index


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    index: dict,
    top_k: int = 5,
    filter_type: str = "all"    # "all" | "text" | "image"
) -> list[dict]:
   
    embeddings = index["embeddings"]
    metadata   = index["metadata"]
    use_clip   = index.get("use_clip_for_text", False)

    if use_clip:
        query_vec = embed_texts_clip([query])[0]   # shape (512,)
    else:
        query_vec = embed_texts([query])[0]         # shape (384,)

    # Filter to matching type and matching embedding dimension
    query_dim = query_vec.shape[0]

    selected_indices = []
    for i, meta in enumerate(metadata):
        if filter_type != "all" and meta["type"] != filter_type:
            continue
        if embeddings[i].shape[0] != query_dim:
            continue  # skip mismatched dims (e.g. CLIP image vs MiniLM text)
        selected_indices.append(i)

    if not selected_indices:
        return []

    subset = embeddings[selected_indices]                 # (M, dim)
    scores = subset @ query_vec                           # cosine sim = dot product (both normalised)

    top_local = np.argsort(scores)[::-1][:top_k]
    results = []
    for local_i in top_local:
        global_i = selected_indices[local_i]
        entry = dict(metadata[global_i])
        entry["score"] = float(scores[local_i])
        results.append(entry)

    return results


# ── Persistence ───────────────────────────────────────────────────────────────

def save_index(index: dict, path: str = "index.pkl") -> None:
    with open(path, "wb") as f:
        pickle.dump(index, f)
    print(f"Index saved to {path}")


def load_index(path: str = "index.pkl") -> dict:
    with open(path, "rb") as f:
        index = pickle.load(f)
    print(f"Index loaded: {index['embeddings'].shape[0]} entries")
    return index

