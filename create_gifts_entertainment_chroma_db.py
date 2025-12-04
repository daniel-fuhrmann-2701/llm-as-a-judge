import logging
from pathlib import Path
from typing import List

# LangChain imports
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader
)

# Document processing
import pandas as pd


def load_documents(data_dir: Path) -> List[Document]:
    """Load and process documents from the given directory and all subdirectories."""
    documents = []
    
    # Supported file extensions
    exts = [".pdf", ".docx", ".doc", ".txt", ".xlsx", ".xls"]
    
    # Get all files recursively from all subdirectories
    file_paths = [p for p in data_dir.rglob("*") if p.suffix.lower() in exts]
    
    print(f"Found {len(file_paths)} documents to process across all subdirectories")
    print(f"Scanning directory: {data_dir}")
    
    # Group files by subdirectory for better logging
    subdirs = {}
    for file_path in file_paths:
        subdir = file_path.parent.name
        if subdir not in subdirs:
            subdirs[subdir] = []
        subdirs[subdir].append(file_path)
    
    print(f"Found documents in {len(subdirs)} subdirectories:")
    for subdir, files in subdirs.items():
        print(f"  - {subdir}: {len(files)} files")
    
    processed_count = 0
    for file_path in file_paths:
        try:
            suffix = file_path.suffix.lower()
            
            if suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
            elif suffix in [".docx", ".doc"]:
                loader = Docx2txtLoader(str(file_path))
                docs = loader.load()
            elif suffix == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()
            elif suffix in [".xlsx", ".xls"]:
                # Process Excel files
                df = pd.read_excel(file_path)
                content = df.to_string(index=False)
                docs = [Document(page_content=content, metadata={"source": str(file_path)})]
            else:
                continue
            
            # Add metadata for gifts & entertainment context
            for doc in docs:
                doc.metadata["subdirectory"] = file_path.parent.name
                doc.metadata["file_type"] = suffix
                doc.metadata["document_category"] = "gifts_entertainment"
                doc.metadata["source_system"] = "berenberg_gifts_entertainment"
                
            documents.extend(docs)
            processed_count += 1
            print(f"Processed ({processed_count}/{len(file_paths)}): {file_path.parent.name}/{file_path.name}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return documents


def main():
    logging.basicConfig(level=logging.INFO)
    
    # Source document folder for gifts & entertainment data
    data_dir = Path(r"W:\T1368-WAM AI\03 Projects\02 AI Projects\01 Berenberg AI Hub\17 Gifts & Entertainment data")
    if not data_dir.exists():
        print(f"❌ Source directory not found: {data_dir}")
        return
    
    # Output directory for Chroma DB - same location as other ChromaDBs
    output_dir = Path(r"\\dnsbego.de\dfsbego\home04\FuhrmannD\Documents\01_Trainee\Master\Thesis\code\agentic_system\data")
    chroma_dir = output_dir / "gifts_entertainment_chroma_db"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Creating Gifts & Entertainment ChromaDB ===")
    print(f"Source directory: {data_dir}")
    print(f"Output directory: {chroma_dir}")
    print()
    
    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Load documents
    print("Loading gifts & entertainment documents...")
    documents = load_documents(data_dir)
    
    if not documents:
        print("❌ No documents found to process.")
        return
    
    # Split documents into chunks
    print(f"Splitting {len(documents)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create Chroma vector store
    print(f"Creating ChromaDB with {len(chunks)} chunks...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="berenberg_gifts_entertainment",
        persist_directory=str(chroma_dir)
    )
    
    print("✅ Gifts & Entertainment ChromaDB created successfully!")
    print(f"   Location: {chroma_dir}")
    print(f"   Total documents: {len(documents)}")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Collection name: berenberg_gifts_entertainment")
    print()
    
    print("=== Summary ===")
    print("✅ Gifts & Entertainment ChromaDB creation completed!")
    print(f"📁 Database location: {chroma_dir}")
    print(f"📊 Collection: berenberg_gifts_entertainment")


if __name__ == "__main__":
    main()