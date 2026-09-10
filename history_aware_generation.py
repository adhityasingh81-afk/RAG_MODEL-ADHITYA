from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================================================
# 1. Connect to ChromaDB
# =========================================================

persistent_directory = "db/chroma_db"


# IMPORTANT:
# This must be the SAME embedding model used during ingestion.
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)


# =========================================================
# 2. Load Local Hugging Face Language Model
# =========================================================

print("Loading Hugging Face language model...")

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name
)


# =========================================================
# 3. Function to Generate Text
# =========================================================

def generate_text(prompt, max_new_tokens=150):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return generated_text.strip()


# =========================================================
# 4. Conversation History
# =========================================================

chat_history = []


# =========================================================
# 5. Ask Question
# =========================================================

def ask_question(user_question):

    print(f"\n--- You asked: {user_question} ---")


    # =====================================================
    # STEP 1: Make the question standalone
    # =====================================================

    if chat_history:

        history_text = "\n".join(
            [
                f"User: {message.content}"
                if isinstance(message, HumanMessage)
                else f"Assistant: {message.content}"
                for message in chat_history
            ]
        )

        rewrite_prompt = f"""
Given the conversation history below, rewrite the user's
new question so that it is completely standalone and
easy to search in a document database.

Conversation history:
{history_text}

New question:
{user_question}

Return ONLY the rewritten question.
"""

        search_question = generate_text(
            rewrite_prompt,
            max_new_tokens=80
        )

        print(f"Searching for: {search_question}")

    else:

        search_question = user_question


    # =====================================================
    # STEP 2: Find relevant documents
    # =====================================================

    retriever = db.as_retriever(
        search_kwargs={"k": 3}
    )

    docs = retriever.invoke(search_question)


    print(f"Found {len(docs)} relevant documents:")

    for i, doc in enumerate(docs, 1):

        lines = doc.page_content.split("\n")[:2]

        preview = "\n".join(lines)

        print(f"  Doc {i}: {preview}...")


    # =====================================================
    # STEP 3: Prepare document context
    # =====================================================

    context = "\n\n".join(
        [
            f"Document {i}:\n{doc.page_content}"
            for i, doc in enumerate(docs, 1)
        ]
    )


    # =====================================================
    # STEP 4: Prepare conversation history
    # =====================================================

    history_text = "\n".join(
        [
            f"User: {message.content}"
            if isinstance(message, HumanMessage)
            else f"Assistant: {message.content}"
            for message in chat_history
        ]
    )


    # =====================================================
    # STEP 5: Create final RAG prompt
    # =====================================================

    combined_input = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the information
contained in the documents below.

Conversation history:
{history_text}

Documents:
{context}

Current question:
{user_question}

Rules:
1. Answer clearly and concisely.
2. Use only information from the documents.
3. Use the conversation history only to understand the question.
4. Do not make up information.
5. If the documents do not contain enough information, say:
"I don't have enough information to answer that question
based on the provided documents."
"""


    # =====================================================
    # STEP 6: Generate final answer using FLAN-T5
    # =====================================================

    answer = generate_text(
        combined_input,
        max_new_tokens=150
    )


    # =====================================================
    # STEP 7: Save conversation
    # =====================================================

    chat_history.append(
        HumanMessage(content=user_question)
    )

    chat_history.append(
        AIMessage(content=answer)
    )


    # =====================================================
    # STEP 8: Display answer
    # =====================================================

    print(f"Answer: {answer}")

    return answer


# =========================================================
# 6. Start Chat
# =========================================================

def start_chat():

    print("Ask me questions! Type 'quit' to exit.")

    while True:

        question = input("\nYour question: ")

        if question.lower() == "quit":

            print("Goodbye!")

            break

        ask_question(question)


# =========================================================
# 7. Main
# =========================================================

if __name__ == "__main__":
    start_chat()