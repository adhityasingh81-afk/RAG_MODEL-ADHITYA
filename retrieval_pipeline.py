from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# Path where ChromaDB was saved
persist_directory = "db/chroma_db"


# Use the SAME embedding model used during ingestion
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


# Load the existing ChromaDB
db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)


# Question to search
query = "What was NVIDIA's first graphics accelerator called?"


# Create retriever
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)


# Retrieve relevant documents
relevant_docs = retriever.invoke(query)


# Display results
print(f"\nUser Query: {query}")
print("\n--- Retrieved Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"\nDocument {i}:")
    print(doc.page_content)
    print("-" * 80)