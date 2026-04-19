# 📄 PDF QA Bot

A production-grade AI-powered chatbot that lets you upload any PDF and ask questions about it in natural language. Built with a full RAG (Retrieval-Augmented Generation) pipeline using CLIP, FAISS, and Groq.

🚀 **[Live Demo](https://huggingface.co/spaces/Sanaashraf/Pdf-QA-bot)**

---

## ✨ Features

- 📤 Upload any text-based PDF
- 💬 Ask questions in natural language — get answers with **page citations**
- 🖼️ Image-aware retrieval — finds relevant diagrams and figures from the PDF
- 🔁 Multi-turn conversation with memory
- ⚙️ Adjustable settings — model, chunk count, similarity threshold
- ⚡ Powered by Groq (llama-3.3-70b) for fast responses

---

## 🏗️ Architecture

```
PDF File
   │
   ▼
pdf_loader.py      ← PyMuPDF extracts text + images per page
   │
   ▼
chunking.py        ← LangChain splits text into overlapping chunks
   │
   ▼
embeddings.py      ← CLIP / MiniLM converts chunks → vectors
   │
   ▼
vector_store.py    ← FAISS indexes vectors for fast similarity search
   │
   ▼
qa_chain.py        ← Query → retrieve chunks → Groq LLM → answer
   │
   ▼
app.py             ← Streamlit UI
```

This is a two-phase pipeline:

| Phase | When | Steps |
|-------|------|-------|
| **Indexing** | Once, on PDF upload | Load → Chunk → Embed → Store |
| **Querying** | Every question | Embed query → Search → LLM → Answer |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| PDF Extraction | PyMuPDF (fitz) |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Text Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Image Embeddings | openai/clip-vit-base-patch32 |
| Vector Search | FAISS (IndexFlatIP) |
| LLM | Groq — llama-3.3-70b-versatile |
| Deployment | HuggingFace Spaces |

---

## 📁 Project Structure

```
Pdf-QA-bot/
├── app.py
├── requirements.txt
├── packages.txt
├── README.md
├── .gitignore
└── utils/
    ├── pdf_loader.py
    ├── chunking.py
    ├── embeddings.py
    ├── vector_store.py
    └── qa_chain.py
```

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/sana-ash110/Pdf-QA-bot.git
cd Pdf-QA-bot
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Groq API key**

Create a `.env` file in the project root:
```
GROQ_API_KEY=gsk_your_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

**5. Run the app**
```bash
streamlit run app.py
```

---

## 🔧 How It Works

### PDF Loading
PyMuPDF (`fitz`) opens the PDF and extracts plain text and embedded images from each page. Images smaller than 100×100px are skipped (decorative elements). Each page returns a dict with `page_number`, `text`, `source`, and `images`.

### Chunking
Text is split using LangChain's `RecursiveCharacterTextSplitter` with `chunk_size=500` and `overlap=100`. The overlap ensures sentences near chunk boundaries appear in both adjacent chunks so they're never missed during retrieval.

### Embeddings
Two models are used:
- **CLIP ViT-B/32** — embeds both text and images into the same 512-dimensional vector space, enabling cross-modal retrieval (text query → find relevant image)
- **MiniLM-L6-v2** — faster 384-dim text-only embeddings when image retrieval isn't needed

### Vector Search
All embeddings are stored in a **FAISS IndexFlatIP** (exact inner product search). Since vectors are L2-normalised, inner product equals cosine similarity. At query time, the user's question is embedded and the top-k most similar chunks are retrieved.

### RAG Chain
Retrieved chunks are injected into a Groq prompt as context. The LLM is instructed to answer only from the provided context and cite page numbers. Conversation history (last 10 turns) is maintained for follow-up questions.

---

## ⚙️ Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Groq model | llama-3.3-70b-versatile | Switch to llama3-8b for faster responses |
| top_k | 5 | Number of chunks retrieved per question |
| min_score | 0.3 | Minimum cosine similarity to include a chunk |
| Use CLIP for text | On | Enables cross-modal image retrieval |

---

## 🌐 Deploy on HuggingFace Spaces

1. Fork this repo or create a new Space
2. Upload all files maintaining the `src/` folder structure
3. Go to **Settings → Repository secrets** and add:
   ```
   GROQ_API_KEY = gsk_your_key_here
   ```
4. The Space builds automatically — first build takes ~5 minutes (model downloads)

---

## 📌 Known Limitations

- Works best with text-based PDFs — scanned/image-only PDFs won't extract text
- CLIP token limit is 77 tokens — very long chunk text is truncated during embedding
- HF Spaces free tier sleeps after 48h of inactivity (30s cold start on wake)

---

## 🔮 Possible Improvements

- [ ] Multi-PDF support with per-document filtering
- [ ] Streaming responses with `st.write_stream()`
- [ ] Hybrid retrieval (BM25 + dense) with Reciprocal Rank Fusion
- [ ] Cross-encoder re-ranking for better answer quality
- [ ] RAG evaluation with RAGAs framework
- [ ] Voice input/output with Whisper + TTS

---

## 👩‍💻 Author

**Sana Ashraf**
Final year BS Information Technology — PUCIT, Lahore

[![GitHub](https://img.shields.io/badge/GitHub-sana--ash110-181717?logo=github)](https://github.com/sana-ash110)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sana%20Ashraf-0A66C2?logo=linkedin)](https://linkedin.com/in/sana-ashraf-24a9b7302)

---

## 📄 License

MIT License — feel free to use, modify, and build on this project.
