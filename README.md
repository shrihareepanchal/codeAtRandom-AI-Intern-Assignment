#  Multi-Document Embedding Search Engine with Caching

##  1. Folder Structure

```
multi_doc_search_full/
├─ src/
│  ├─ config.py                # Configuration settings
│  ├─ embedder.py              # Embedding model wrapper
│  ├─ cache_manager.py         # SQLite caching system
│  ├─ search_engine.py         # Core search & ranking logic
│  ├─ indexing.py              # Index builder
│  ├─ api.py                   # FastAPI application
│  └─ utils.py                 # Helper functions
├─ scripts/
│  └─ preprocess_and_index.py  # Build embeddings + index
├─ data/
│  └─ docs/                    # Folder containing .txt documents
├─ cache/                      # Stores FAISS index & SQLite DB
├─ download_dataset.py         # Used to download and generate dataset (.txt files)
├─ requirements.txt
├─ README.md
└─ .gitignore
```

---

##  2. How Caching Works

1. Each `.txt` document in `data/docs/` is:
   - Cleaned (lowercased, whitespace normalized, HTML removed)
   - Hashed using **SHA-256** to generate a unique signature
2. This hash and embedding are stored in **SQLite (`cache/embeddings.db`)**
3. If a document remains unchanged, the embedding is reused from cache
4. Only new or modified files are re-embedded, significantly improving performance

 Cache fields:
```
doc_id, filename, content_hash, embedding, doc_length, updated_at
```

---

##  3. How to Run Embedding Generation

###  Step 1 — Generate Dataset (If Needed)

```bash
python download_dataset.py
```

This will create 200 `.txt` files in `data/docs/`.

---

###  Step 2 — Build Index & Generate Embeddings

```bash
python scripts/preprocess_and_index.py
```

Expected output:
```
Building index over documents in ./data/docs ...
Indexed <X> documents.
Done.
```

---

##  4. How to Start API

```bash
uvicorn src.api:app --reload
```

Then open:
  http://127.0.0.1:8000/docs

Example request:
```json
{
  "query": "machine learning",
  "top_k": 5
}
```

---

##  5. Design Choices

| Component | Decision | Reason |
|-----------|----------|--------|
| Embedding Model | `all-MiniLM-L6-v2` (384 dims) | Fast & accurate |
| Caching | SQLite + SHA-256 | Avoids redundant computation |
| Search Engine | FAISS (IP) / NumPy fallback | Efficient similarity search |
| API Framework | FastAPI | Lightweight, modern |
| File Format | `.txt` per document | Simple & scalable |
| Ranking | Frequency + normalization | Human-interpretable |
| Preprocessing | Lowercase, clean spaces | Improve embedding results |

---

##  Example API Response

```json
{
  "results": [
    {
      "doc_id": "doc_83.txt",
      "score": 0.2191,
      "preview": "digital imaging computing techniques...",
      "explanation": {
        "matched_keywords": ["computing"],
        "overlap_ratio": 0.5,
        "length_normalization": 0.137,
        "query_keywords": ["digital", "computing"]
      }
    }
  ]
}
```

---

##  Quick Run Guide

```bash
# 1. Optional – Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Generate dataset
python download_dataset.py

# 4. Build embeddings and index
python scripts/preprocess_and_index.py

# 5. Start API
uvicorn src.api:app --reload

# 6. Open
http://127.0.0.1:8000/docs
```
