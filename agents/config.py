"""
Configuration settings for Clinical Review Assistant
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_SUMMARY = "gpt-4o"
MODEL_EVALUATION = "gpt-4o"
TEMPERATURE_SUMMARY = 0.3
TEMPERATURE_EVALUATION = 0.1
MAX_TOKENS_SUMMARY = 2000
MAX_TOKENS_EVALUATION = 1500


# Embedding Configuration
EMBEDDING_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# PolicyMind Configuration
MAX_CANDIDATES = 100
RATE_LIMIT_TURNS = 20

CHROMA_PERSIST_DIR = "./data/embeddings"
COLLECTION_NAME = "medical_policies"
TOP_K_POLICIES = 5
TOP_K_CHUNKS = 10  # For Node 3 RAG retrieval

# PolicyMind Data Ingestion Paths
BASE_DIR = Path("./data")
POLICY_INPUT_DIR = BASE_DIR / "raw_policy_pdf"
POLICY_OUTPUT_DIR = BASE_DIR / "policy_txt"
PERSIST_DIR = BASE_DIR / "embeddings"

# Pinecone Configuration for PolicyMind
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-policies")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "policy_docs")

# PolicyMind Chunking (for policy documents - section-aware)
POLICY_CHUNK_SIZE = 800
POLICY_CHUNK_OVERLAP = 100

# Security Configuration
MAX_FILE_SIZE_MB = 50
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60

# Streamlit Configuration
PAGE_TITLE = "Clinical Review Assistant"
PAGE_ICON = "🏥"
LAYOUT = "wide"

# Agent Configuration
MAX_AGENT_ITERATIONS = 10

# Authentication Configuration
USERS_FILE = "users.json"

# ChromaDB Configuration for Node 1 & 3 (Ephemeral)
EPHEMERAL_COLLECTION_NAME = "medical_records_ephemeral"
