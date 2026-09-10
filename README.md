# RAG Model - Ingestion Pipeline

A Retrieval-Augmented Generation (RAG) document ingestion pipeline built with LangChain, ChromaDB, and HuggingFace Sentence Transformers.

## Features

- **Document Loading**: Loads text files (`.txt`) from the `docs/` directory with UTF-8 encoding support via `DirectoryLoader` and `TextLoader`.
- **Text Chunking**: Splits documents into manageable chunks with character text splitting (`CharacterTextSplitter`).
- **Embedding Generation**: Uses the open-source HuggingFace model `sentence-transformers/all-MiniLM-L6-v2` running locally on CPU.
- **Vector Storage**: Persists document vectors into a local ChromaDB collection (`db/chroma_db`) with cosine similarity indexing.

## Project Structure

```
.
├── docs/                     # Source documents (.txt files)
├── ingestion_pipeline.py     # Document loading, chunking, and embedding pipeline
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
└── README.md
```

## Setup & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/adhityasingh81-afk/RAG_MODEL-ADHITYA.git
   cd RAG_MODEL-ADHITYA
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add documents**:
   Place your `.txt` files in the `docs/` directory.

5. **Run the ingestion pipeline**:
   ```bash
   python ingestion_pipeline.py
   ```
