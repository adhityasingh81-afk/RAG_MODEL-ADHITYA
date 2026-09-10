from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os


def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs = {"encoding":"utf-8"} 
        )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
   
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents 


def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """Split documents into smaller chunks and remove exact duplicates."""

    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    # Remove exact duplicate chunks
    unique_chunks = []
    seen = set()

    for chunk in chunks:
        text = chunk.page_content.strip()

        if not text:
            continue

        if text not in seen:
            seen.add(text)
            unique_chunks.append(chunk)

    print(f"Original chunks: {len(chunks)}")
    print(f"Unique chunks: {len(unique_chunks)}")
    print(f"Duplicates removed: {len(chunks) - len(unique_chunks)}")

    if unique_chunks:
        for i, chunk in enumerate(unique_chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata.get('source')}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)

    if len(unique_chunks) > 5:
        print(f"\n... and {len(unique_chunks) - 5} more chunks")

    return unique_chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""

    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("--- Creating vector store ---")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("--- Finished creating vector store ---")
    print(f"Vector store created and saved to {persist_directory}")

    return vectorstore


def main () :
    print("Main Function")
    
    #loading the documents 
    documents = load_documents (docs_path= "docs")

    #chunking the loaded documents 
    chunks = split_documents(documents)

    #Embedding and storing the chunks 
    vectorstore = create_vector_store(chunks)



if __name__ == "__main__":
    main () 
   