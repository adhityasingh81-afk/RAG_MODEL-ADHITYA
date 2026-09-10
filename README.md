# Self-Learning AI - RAG & Conversational Generation

A Retrieval-Augmented Generation (RAG) and conversational AI system built with LangChain, ChromaDB, HuggingFace Sentence Transformers, and Google FLAN-T5.

## Features

- **Document Loading & Chunking**: Loads text files (`.txt`) from `docs/` with UTF-8 support and deduplicated chunking.
- **Embedding Generation**: Uses `sentence-transformers/all-MiniLM-L6-v2` running locally on CPU.
- **Vector Storage**: Persists document vectors into local ChromaDB collections (`db/chroma_db`) with cosine similarity indexing.
- **Similarity Retrieval**: Fast query-based retrieval of relevant document context.
- **Answer Generation**: Synthesizes grounded answers using local `google/flan-t5-base`.
- **History-Aware Conversational QA**: Rewrites follow-up questions to be standalone and maintains chat history for contextual conversations.

## Project Structure

```
.
├── docs/                           # Source documents (.txt files)
├── ingestion_pipeline.py           # Document loading, deduplicating, and ChromaDB vector store creation
├── retrieval_pipeline.py           # Semantic search and document context retrieval
├── answer_generation.py            # RAG QA pipeline powered by FLAN-T5
├── history_aware_generation.py     # Conversational history-aware RAG pipeline
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
└── README.md
```

## Setup & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/adhityasingh81-afk/self-learning-ai.git
   cd self-learning-ai
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

6. **Run conversational QA**:
   ```bash
   python history_aware_generation.py
   ```

