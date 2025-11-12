from langchain_core.documents import Document
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from collections import defaultdict
import warnings, os, sys, re
from pinecone import Pinecone, ServerlessSpec
import time
import hashlib

from agents.config import (
    POLICY_INPUT_DIR, POLICY_OUTPUT_DIR, EMBEDDING_MODEL,
    POLICY_CHUNK_SIZE, POLICY_CHUNK_OVERLAP,
    PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_NAMESPACE
)

# Suppress all warnings
warnings.filterwarnings("ignore")
sys.stderr = open(os.devnull, "w")

CHUNK_SIZE = POLICY_CHUNK_SIZE
CHUNK_OVERLAP = POLICY_CHUNK_OVERLAP
POLICY_EMBED_MODEL = EMBEDDING_MODEL

# Create directories if they don't exist
for directory in [POLICY_INPUT_DIR, POLICY_OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def sanitize_vector_id(policy_id, index):
    """
    Create ASCII-safe vector ID for Pinecone
    Replaces non-alphanumeric characters with underscores
    
    Args:
        policy_id: Original policy identifier
        index: Document index
        
    Returns:
        ASCII-safe vector ID
    """
    # Replace non-alphanumeric characters with underscore
    clean_id = re.sub(r'[^a-zA-Z0-9]', '_', str(policy_id))
    # Remove consecutive underscores
    clean_id = re.sub(r'_+', '_', clean_id)
    # Remove leading/trailing underscores
    clean_id = clean_id.strip('_')
    # Combine with index
    return f"{clean_id}_{index}"

def extract_pdf_with_pages(pdf_path):
    """
    Extract PDF content with page tracking using UnstructuredPDFLoader
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dictionary mapping page numbers to text content
    """
    print(f"   Loading: {pdf_path.name}")
    
    # Load with mode="elements" to get page tracking
    loader = UnstructuredPDFLoader(str(pdf_path), mode="elements")
    elements = loader.load()
    
    # Group elements by page number
    pages_dict = defaultdict(list)
    for elem in elements:
        # Skip elements with no content
        if elem.page_content is None or not str(elem.page_content).strip():
            continue
            
        page_num = elem.metadata.get("page_number", 1)
        pages_dict[page_num].append(str(elem.page_content))
    
    # Combine elements within each page
    pages_text = {}
    for page_num, texts in pages_dict.items():
        combined = "\n".join(texts)
        if combined.strip():  # Only add non-empty pages
            pages_text[page_num] = combined
    
    print(f"Extracted {len(pages_text)} pages")
    return pages_text

def chunk_policy_text_with_pages(pages_text_dict, policy_id, chunk_size=None, chunk_overlap=None):
    """
    Chunk policy text with regex-based section detection while preserving page numbers
    INCLUDES FALLBACK: If no sections found, chunks entire page content
    
    Args:
        pages_text_dict: Dict mapping page numbers to text
        policy_id: Policy identifier
        chunk_size: Size of chunks (uses config if None)
        chunk_overlap: Overlap between chunks (uses config if None)
        
    Returns:
        List of Document objects with page numbers
    """
    chunk_size = chunk_size or POLICY_CHUNK_SIZE
    chunk_overlap = chunk_overlap or POLICY_CHUNK_OVERLAP
    # Regex pattern for policy section headers
    section_pattern = re.compile(
        r"(?m)(?:^|\n)([A-Za-z][A-Za-z\s]{3,}?"
        r"(?:Description|Medically Necessary|Not Medically Necessary|Coding|Discussion/General Information|Definitions|References|"
        r"Coverage Guidance|Summary of Evidence|Analysis of Evidence|Rationale for Determination|General Information|Revision History Information|Keywords|"
        r"Covered Indications|Limitations|History|Scope|Indications and Limitations of Coverage)[:]?)",
        flags=re.IGNORECASE
    )
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    all_chunks = []
    
    # Process each page
    for page_num in sorted(pages_text_dict.keys()):
        page_text = pages_text_dict[page_num]
        
        # Skip None or empty pages
        if page_text is None or not str(page_text).strip():
            continue
        page_text = str(page_text).strip()

        if not page_text:
            continue

        # Try regex-based section detection
        sections = re.split(section_pattern, page_text)
        
              # Check if sections were found
        if len(sections) > 1:
            # Sections found - process with headers
            for i in range(1, len(sections), 2):
                # Check if sections[i] exists and is not None
                if i >= len(sections) or sections[i] is None:
                    continue
                
                header = str(sections[i]).strip().title() if sections[i] else "Content"
                # Check if sections[i+1] exists and is not None
                if i + 1 >= len(sections) or sections[i + 1] is None:
                    content = ""
                else:
                    content = str(sections[i + 1]).strip()
                
                if not content:
                    continue
                
                # Apply chunking within section
                sub_chunks = splitter.split_text(content)
                
                for j, chunk_text in enumerate(sub_chunks):
                    #Skip empty chunks
                    if not chunk_text or not chunk_text.strip():
                        continue
                    
                    doc = Document(
                        page_content=f"{header}\n{chunk_text.strip()}",
                        metadata={
                            "policy_id": policy_id,
                            "source": f"{policy_id}.pdf",
                            "page": page_num, 
                            "section": header,
                            "chunk_index": j
                        }
                    )
                    all_chunks.append(doc)
        else:
            # NO sections found - fallback to chunking entire page
            page_chunks = splitter.split_text(page_text)
            
            for j, chunk_text in enumerate(page_chunks):
                # Skip None or empty chunks
                if not chunk_text or not str(chunk_text).strip():
                    continue
                
                doc = Document(
                    page_content=str(chunk_text).strip(),
                    metadata={
                        "policy_id": policy_id,
                        "source": f"{policy_id}.pdf",
                        "page": page_num,
                        "section": "Full Page Content",
                        "chunk_index": j
                    }
                )
                all_chunks.append(doc)
    
    return all_chunks

def save_text_files():
    """
    Optional: Save extracted text to .txt files for inspection
    """
    pdf_files = list(POLICY_INPUT_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"\nNo PDF files found in {POLICY_INPUT_DIR}")
        return
    
    print(f"\nConverting {len(pdf_files)} PDFs to text files...")
    
    for pdf_file in pdf_files:
        try:
            pages_dict = extract_pdf_with_pages(pdf_file)

            if not pages_dict:  # Check if any pages extracted
                print(f"Â Skipped {pdf_file.name}: No content extracted")
                continue
            
            # Combine all pages with page markers
            full_text = ""
            for page_num in sorted(pages_dict.keys()):
                page_content = pages_dict[page_num]
                if page_content is None or not str(page_content).strip():
                    continue
                    
                full_text += f"\n\n{'='*50}\n"
                full_text += f"PAGE {page_num}\n"
                full_text += f"{'='*50}\n\n"
                full_text += str(page_content)
            
            if not full_text.strip():
                print(f"  Skipped {pdf_file.name}: No text content")
                continue
            
            # Save to text file
            output_path = POLICY_OUTPUT_DIR / f"{pdf_file.stem}.txt"
            output_path.write_text(full_text.strip(), encoding="utf-8")
            print(f"   Saved: {output_path.name}")
            
        except Exception as e:
            print(f" Failed: {pdf_file.name} - {e}")
    
    print(f"\nText files saved to: {POLICY_OUTPUT_DIR}")

# Database Creation- Ingestion Pipeline
def build_vector_database():
    """
    Build ChromaDB vector database from policy PDFs with actual page numbers
    """
    pdf_files = list(POLICY_INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"\nNo PDF files found in {POLICY_INPUT_DIR}")
        print(f"   Please add policy PDF files to: {POLICY_INPUT_DIR.absolute()}")
        return
    
    print(f"\nProcessing {len(pdf_files)} policy documents...")
    all_documents = []
    
    # Process each PDF
    for pdf_file in pdf_files:
        try:
            policy_id = pdf_file.stem
            print(f"\n{policy_id}")
            
            # Extract pages with tracking
            pages_dict = extract_pdf_with_pages(pdf_file)
            if not pages_dict:
                print(f"  Skipped: No pages extracted")
                continue

            # Chunk with page preservation
            chunks = chunk_policy_text_with_pages(pages_dict, policy_id)
            if not chunks:
                print(f" Skipped: No chunks created")
                continue

            print(f" Created {len(chunks)} chunks with page tracking")
            all_documents.extend(chunks)
            
        except Exception as e:
            print(f"  Error processing {pdf_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_documents:
        print("\n No documents to embed. Check PDF files and try again.")
        return
    

    print(f"Total chunks created: {len(all_documents)}")
    print(f"From {len(pdf_files)} policy documents")
    
    # Create embeddings
    print(f"\nCreating embeddings with {EMBEDDING_MODEL}...")
    print("   This may take a few minutes...")
    
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Initialize Pinecone
    print(f"\nInitializing Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists, if not create it
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    # Get embedding dimension
    sample_embedding = embedder.embed_query("test")
    dimension = len(sample_embedding)
    
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        # Wait for index to be ready
        time.sleep(1)
    else:
        print(f"Using existing Pinecone index: {PINECONE_INDEX_NAME}")
    
    # Get index
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # Prepare vectors for upsert
    print(f"\nUpserting vectors to Pinecone...")
    vectors_to_upsert = []
    batch_size = 100
    
    for i, doc in enumerate(all_documents):
        try:
            # Create embedding
            embedding = embedder.embed_query(doc.page_content)
            
            # Prepare metadata (Pinecone has limitations on metadata size)
            metadata = {
                "policy_id": doc.metadata.get('policy_id', ''),
                "source": doc.metadata.get('source', ''),
                "page": int(doc.metadata.get('page', 1)),
                "section": doc.metadata.get('section', '')[:200],  # Limit size
                "text": doc.page_content[:1000]  # Store truncated text for retrieval
            }
            
            vectors_to_upsert.append({
                "id": sanitize_vector_id(doc.metadata.get('policy_id', 'doc'), i),
                "values": embedding,
                "metadata": metadata
            })
            
            # Upsert in batches
            if len(vectors_to_upsert) >= batch_size:
                retry_count = 0
                max_retries = 3
                
                while retry_count < max_retries:
                    try:
                        index.upsert(vectors=vectors_to_upsert, namespace=PINECONE_NAMESPACE)
                        print(f"   Upserted {i+1}/{len(all_documents)} vectors...")
                        vectors_to_upsert = []
                        time.sleep(0.5)  # Rate limiting delay
                        break
                    except Exception as upsert_error:
                        retry_count += 1
                        print(f"   Retry {retry_count}/{max_retries} after error: {upsert_error}")
                        time.sleep(2 * retry_count)  # Exponential backoff
                        if retry_count >= max_retries:
                            raise
        
        except Exception as e:
            print(f"   Error processing document {i}: {e}")
            continue
    
    # Upsert remaining vectors
    if vectors_to_upsert:
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                index.upsert(vectors=vectors_to_upsert, namespace=PINECONE_NAMESPACE)
                print(f"   Upserted final batch: {len(all_documents)} total vectors")
                break
            except Exception as upsert_error:
                retry_count += 1
                print(f"   Retry {retry_count}/{max_retries} for final batch: {upsert_error}")
                time.sleep(2 * retry_count)
                if retry_count >= max_retries:
                    print(f"   Warning: Failed to upsert final batch after {max_retries} retries")
    
    # Verify
    stats = index.describe_index_stats()
    count = stats.namespaces.get(PINECONE_NAMESPACE, {}).get('vector_count', 0)
    
    print(f"\nSUCCESS!")
    print(f"   Documents embedded: {count}")
    print(f"   Index: {PINECONE_INDEX_NAME}")
    print(f"   Namespace: {PINECONE_NAMESPACE}")

    
    # Show sample metadata
    if all_documents:
        sample = all_documents[0]
        print(f"\nSample chunk metadata:")
        print(f"   Policy ID: {sample.metadata.get('policy_id')}")
        print(f"   Source: {sample.metadata.get('source')}")
        print(f"   Page: {sample.metadata.get('page')}")
        print(f"   Section: {sample.metadata.get('section')}")


# Main execution
if __name__ == "__main__":
    print("\nPolicyMind Data Ingestion Starting...\n")
    
    # Step 1:  Saving text files for inspection
    save_text_files()
    
    # Step 2: Build vector database with page tracking
    build_vector_database()

    print("\nData ingestion complete!")
    print(f"\nNext steps:")
    print(f"1. Verify Pinecone index: {PINECONE_INDEX_NAME}")
    print(f"2. Run your CRA app: streamlit run app.py")