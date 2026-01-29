"""
RAG Server Configuration

All settings can be overridden via environment variables.
"""
import os

# Import for HuggingFaceEmbeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback for backward compatibility
    from langchain_community.embeddings import HuggingFaceEmbeddings

# =============================================================================
# PATH SETTINGS
# =============================================================================
DATA_DIR = "/app/data"
VECTOR_DB_DIR = os.path.join(DATA_DIR, "vector_db")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
MODELS_DIR = os.path.join(DATA_DIR, "models")

# =============================================================================
# EMBEDDING MODEL CONFIGURATION
# =============================================================================
# Configurable via EMBEDDING_MODEL env var
# Recommended models:
#   - all-MiniLM-L6-v2           (EN optimized, 384 dim, fast)
#   - intfloat/multilingual-e5-base (best RU/EN, 768 dim, slower)
#   - paraphrase-multilingual-MiniLM-L12-v2 (multilingual, 384 dim)
#   - cointegrated/rubert-tiny2  (RU only, 312 dim, fastest)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Legacy langchain embeddings (for backward compatibility)
_langchain_model = f"sentence-transformers/{EMBEDDING_MODEL}" if "/" not in EMBEDDING_MODEL else EMBEDDING_MODEL
embeddings = HuggingFaceEmbeddings(model_name=_langchain_model)

# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================
# Configurable via CHUNK_SIZE and CHUNK_OVERLAP env vars
# Recommendations:
#   - Technical docs: CHUNK_SIZE=800, CHUNK_OVERLAP=150
#   - Books/literature: CHUNK_SIZE=1500, CHUNK_OVERLAP=300
#   - Short FAQ: CHUNK_SIZE=500, CHUNK_OVERLAP=100
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# =============================================================================
# CHROMADB SETTINGS
# =============================================================================
CHROMA_SETTINGS = {
    "persist_directory": VECTOR_DB_DIR,
    "anonymized_telemetry": False,
    "allow_reset": True,
    "is_persistent": True
}

# =============================================================================
# MODEL CACHE PATHS (for offline/autonomous operation)
# =============================================================================
# HF_HOME is the primary cache location for HuggingFace (transformers v5+)
os.environ["HF_HOME"] = MODELS_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = MODELS_DIR
os.environ["TORCH_HOME"] = MODELS_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = MODELS_DIR