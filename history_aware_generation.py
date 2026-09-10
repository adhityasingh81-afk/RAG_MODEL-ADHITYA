from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================================================
# 1. ChromaDB Configuration
# =========================================================

persistent_directory = "db/chroma_db"


# =========================================================
# 2. Load Hugging Face Embedding Model
# =========================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


# =========================================================
# 3. Connect to ChromaDB
# =========================================================

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)


# =========================================================
# 4. Load Hugging Face Language Model
# =========================================================

print("Loading Hugging Face language model...")

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
print("Model device:", model.device)

# =========================================================
# 5. Text Generation Function
# =========================================================

def generate_text(prompt, max_new_tokens=80):

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    return generated_text.strip()


# =========================================================
# 6. Conversation History
# =========================================================

chat_history = []


# =========================================================
# 7. Check Whether Question Needs History
# =========================================================

def needs_conversation_history(user_question):

    follow_up_words = [
        "he",
        "she",
        "they",
        "it",
        "this",
        "that",
        "these",
        "those",
        "him",
        "her",
        "them",
        "his",
        "their"
    ]

    words = user_question.lower().split()

    return any(word in words for word in follow_up_words)


# =========================================================
# 8. Ask Question
# =========================================================

def ask_question(user_question):

    print(f"\n--- You asked: {user_question} ---")

    # =====================================================
    # STEP 1: Prepare Search Question
    # =====================================================

    if chat_history and needs_conversation_history(user_question):

        previous_user_question = None

        for message in reversed(chat_history):

            if isinstance(message, HumanMessage):

                previous_user_question = message.content

                break

        if previous_user_question:

            search_question = (
                f"{previous_user_question} {user_question}"
            )

        else:

            search_question = user_question

    else:

        search_question = user_question

    print(f"Searching for: {search_question}")

    # =====================================================
    # STEP 2: Retrieve Documents With Scores
    # =====================================================

    results = db.similarity_search_with_relevance_scores(
        search_question,
        k=3
    )

    print(f"\nFound {len(results)} candidate documents:")

    for i, (doc, score) in enumerate(results, 1):

        print(f"\nDocument {i}")
        print(f"Similarity Score: {score:.4f}")
        print(doc.page_content[:200])

    # =====================================================
    # STEP 3: Filter Weak Documents
    # =====================================================

    relevant_docs = [
        doc
        for doc, score in results
        if score >= 0.40
    ]

    print(f"\nUsing {len(relevant_docs)} relevant documents.")

    # =====================================================
    # STEP 4: No Relevant Documents
    # =====================================================

    if not relevant_docs:

        answer = (
            "I don't have enough information to answer "
            "that question based on the provided documents."
        )

        chat_history.append(
            HumanMessage(content=user_question)
        )

        chat_history.append(
            AIMessage(content=answer)
        )

        print(f"\nAnswer: {answer}")

        return answer

    # =====================================================
    # STEP 5: Prepare Document Context
    # =====================================================

    context = "\n\n".join(
        [
            f"Document {i}:\n{doc.page_content}"
            for i, doc in enumerate(relevant_docs, 1)
        ]
    )

    # =====================================================
    # STEP 6: Prepare Conversation History
    # =====================================================

    recent_history = chat_history[-4:]

    history_text = "\n".join(
        [
            f"User: {message.content}"
            if isinstance(message, HumanMessage)
            else f"Assistant: {message.content}"
            for message in recent_history
        ]
    )

    # =====================================================
    # STEP 7: Create RAG Prompt
    # =====================================================

    combined_input = f"""
You are a question-answering assistant.

Answer the current question using ONLY the information
contained in the documents.

Conversation history:
{history_text}

Documents:
{context}

Current question:
{user_question}

Rules:

1. Answer clearly and concisely.
2. Use ONLY information from the documents.
3. Do NOT use outside knowledge.
4. Do NOT make up information.
5. If the documents do not contain enough information,
say:

I don't have enough information to answer that question
based on the provided documents.
"""

    # =====================================================
    # STEP 8: Generate Answer
    # =====================================================

    import time

    start_time = time.time()

    answer = generate_text(
        combined_input,
        max_new_tokens=80
    )

    end_time = time.time()

    print(
        f"\nGeneration time: "
        f"{end_time - start_time:.2f} seconds"
    )

    # =====================================================
    # STEP 9: Save Conversation
    # =====================================================

    chat_history.append(
        HumanMessage(content=user_question)
    )

    chat_history.append(
        AIMessage(content=answer)
    )

    # =====================================================
    # STEP 10: Display Answer
    # =====================================================

    print(f"\nAnswer: {answer}")

    return answer


# =========================================================
# 9. Start Chat
# =========================================================

def start_chat():

    print("\n==========================================")
    print("       HISTORY-AWARE LOCAL RAG")
    print("==========================================")

    print("Ask me questions!")
    print("Type 'quit' to exit.")


    while True:

        question = input("\nYour question: ")


        if question.lower().strip() == "quit":

            print("Goodbye!")

            break


        if not question.strip():

            print("Please enter a question.")

            continue


        ask_question(question)


# =========================================================
# 10. Main
# =========================================================

if __name__ == "__main__":

    start_chat()