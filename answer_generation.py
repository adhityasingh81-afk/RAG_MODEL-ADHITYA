from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==========================================
# 1. ChromaDB location
# ==========================================

persistent_directory = "db/chroma_db"


# ==========================================
# 2. Load Hugging Face Embedding Model
# ==========================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


# ==========================================
# 3. Load ChromaDB
# ==========================================

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)


# ==========================================
# 4. User Query
# ==========================================

query = "How much did Microsoft pay to acquire GitHub?"


# ==========================================
# 5. Retrieve Relevant Documents
# ==========================================

retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

relevant_docs = retriever.invoke(query)


# ==========================================
# 6. Display Retrieved Context
# ==========================================

print(f"User Query: {query}")

print("\n--- Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:")
    print(doc.page_content)
    print()


# ==========================================
# 7. Combine Query + Retrieved Documents
# ==========================================

context = "\n\n".join(
    [f"Document {i}: {doc.page_content}"
     for i, doc in enumerate(relevant_docs, 1)]
)

combined_input = f"""
Answer the question using ONLY the information provided in the documents.

Question:
{query}

Documents:
{context}

Instructions:
- Give a clear and concise answer.
- Do not use information that is not present in the documents.
- If the documents do not contain enough information, say:
"I don't have enough information to answer that question based on the provided documents."
"""


# ==========================================
# 8. Load Local Hugging Face LLM
# ==========================================

print("\nLoading Hugging Face language model...")

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name
)

inputs = tokenizer(
    combined_input,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False
)

generated_text = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)


# ==========================================
# 10. Display Final Answer
# ==========================================

print("\n--- Generated Response ---")
print(generated_text)