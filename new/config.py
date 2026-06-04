"""
Configuration settings for AI Database Assistant
"""
import os
from pathlib import Path

# Database Configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    ""
)

# Temporary MySQL server used for importing uploaded enterprise dumps/files.
# If you already have a MySQL server running locally, this default usually works.
MYSQL_SERVER_URL = os.getenv(
    "MYSQL_SERVER_URL",
    "mysql+pymysql://root:@localhost:3306/mysql"
)

# LLM Configuration (Ollama)
OLLAMA_BASE_URL = "http://localhost:11434/api"
MODEL_PRESET = os.getenv("MODEL_PRESET", "fast").strip().lower()

FAST_MODELS = {
    "router": "phi3:mini",                 # Intent classification
    "general": "gemma2:2b",                # General non-database answers
    "normalizer": "gemma2:2b",             # Question normalization
    "sql_generator": "qwen2.5-coder:1.5b", # SQL generation
    "sql_repairer": "qwen2.5-coder:1.5b",  # SQL repair
    "humanizer": "gemma2:2b",              # Answer humanization
    "semantic_generator": "gemma2:2b"      # Table semantics
}

BALANCED_MODELS = {
    "router": "phi3:mini",
    "general": "phi3:mini",
    "normalizer": "qwen3:4b",
    "sql_generator": "deepseek-coder:6.7b",
    "sql_repairer": "deepseek-coder:6.7b",
    "humanizer": "phi3:mini",
    "semantic_generator": "qwen3:4b"
}

MODELS = FAST_MODELS if MODEL_PRESET == "fast" else BALANCED_MODELS

# Embedding Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # For all-MiniLM-L6-v2
FAISS_INDEX_TYPE = "IndexFlatL2"
RAG_TOP_K = 3

# Paths
PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "cache"
LOGS_DIR = PROJECT_ROOT / "logs"

CACHE_FILES = {
    "table_semantics": CACHE_DIR / "table_semantics.json",
    "schema_embeddings": CACHE_DIR / "schema_embeddings.npy",
    "index": CACHE_DIR / "faiss.index",
    "schema_metadata": CACHE_DIR / "schema_metadata.json",
    "relationships": CACHE_DIR / "relationships.json",
    "schema_texts": CACHE_DIR / "schema_texts.json",
    "business_rules": CACHE_DIR / "business_rules.json",
    "schema_signature": CACHE_DIR / "schema_signature.json",
    "question_cache": CACHE_DIR / "question_cache.json",
}

# SQL Generation Settings
SQL_MAX_TOKENS = 150
SQL_TEMPERATURE = 0
DEFAULT_LIMIT = 10

# Performance
RESPONSE_TIMEOUT = 10  # seconds target
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "1h")
QUESTION_CACHE_VERSION = 2

# Create directories
CACHE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
