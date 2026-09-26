# NovaCore Multimodal RAG

A Retrieval-Augmented Generation (RAG) pipeline that reads a PDF report — including its **text, tables, and charts/diagrams** — and answers natural-language questions about it, citing page numbers and (when relevant) the original visuals.

## How it works

The pipeline follows the standard RAG stages, extended for multimodal (text + table + image) content:

1. **Load** — open the PDF with PyMuPDF.
2. **Extract** — walk every page and pull out:
   - raw text
   - tables (converted to markdown)
   - images/charts/diagrams
3. **Describe visuals** — each extracted image is sent to a vision LLM (via Groq), which returns a factual text description (key values, trends, diagram flow, etc.).
4. **Chunk** — long text is split into overlapping chunks sized for retrieval; tables and visual descriptions are kept as single units.
5. **Embed** — all chunks are embedded using a Sentence-Transformers model.
6. **Index** — embeddings are upserted into a Pinecone serverless vector index.
7. **Retrieve** — top-k relevant chunks are fetched for a given question.
8. **Generate** — an LLM answers using only the retrieved context:
   - text-only chain if no visuals were retrieved
   - vision-augmented chain (re-attaches the original image) if a chart/diagram was retrieved
9. **Query** — a single `ask_rag()` function ties retrieval + generation together and prints the answer, sources, and (optionally) the retrieved images.

## Tech stack

- **PyMuPDF (`fitz`)** — PDF parsing (text, tables, images)
- **Pandas** — table handling
- **Groq** — LLM inference (text model + vision model)
- **LangChain** — prompt templates, document abstraction, text splitting
- **HuggingFace Sentence-Transformers** — embeddings
- **Pinecone** — vector database

## Setup

### 1. Install dependencies
```bash
pip install -qU langchain langchain-core langchain-groq langchain-huggingface \
    langchain-pinecone langchain-text-splitters pinecone sentence-transformers \
    pymupdf groq pandas tabulate pillow
```

### 2. Set API keys
Never hardcode keys in the script. Set them as environment variables.

**In Google Colab**, store them in Colab's Secrets manager (key icon in the sidebar), then:
```python
from google.colab import userdata
import os

os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
os.environ["PINECONE_API_KEY"] = userdata.get("PINECONE_API_KEY")
```

**Locally**, export them in your shell:
```bash
export GROQ_API_KEY="your-key-here"
export PINECONE_API_KEY="your-key-here"
```

### 3. Configure the PDF path
Update `PDF_PATH` in the script to point to your PDF file.

### 4. Run
```bash
python novacore_multimodal_rag.py
```

## Project structure

```
novacore-multimodal-rag/
├── novacore_multimodal_rag.py   # full pipeline (load → extract → chunk → embed → index → retrieve → generate)
└── README.md
```

## Example query

```python
ask_rag(
    "According to the revenue graph, which quarter had the highest revenue?",
    retriever=retriever,
    show_images=True,
)
```
Output includes the generated answer, the page numbers of the sources used, and (if relevant) the retrieved chart image.

## Notes

- API keys must **never** be committed to the repository. Use environment variables or a secrets manager, and add a `.gitignore` entry for any `.env` file.
- The Pinecone index/namespace is cleared and rebuilt each time the pipeline runs, to keep the demo idempotent — remove that step if you want persistent, incremental indexing.
