"""
Initialization module - runs once at startup.
Loads schema, generates embeddings, builds FAISS index, and reuses cache when possible.
"""
import hashlib
import json
import logging

import faiss
import numpy as np
import pandas as pd
import requests
from sqlalchemy import inspect
from sentence_transformers import SentenceTransformer

from config import (
    CACHE_FILES,
    EMBEDDING_MODEL_NAME,
    MODELS,
    OLLAMA_BASE_URL,
)
from prompts import SEMANTIC_PROMPT
from source_ingestion import build_engine_from_source

logger = logging.getLogger(__name__)


def load_embedding_model():
    """Load the sentence transformer embedding model."""
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def connect_database(database_url=None, uploaded_files=None):
    """Create SQLAlchemy engine from either a connection string or uploaded files."""
    engine, source_info = build_engine_from_source(
        database_url=database_url,
        uploaded_files=uploaded_files,
    )
    logger.info("Database source prepared: %s", source_info.get("kind"))
    return engine, source_info


def extract_schema_metadata(engine):
    """Extract tables, columns, and relationships from the connected database."""
    inspector = inspect(engine)

    tables = inspector.get_table_names()
    views = inspector.get_view_names()

    schema_metadata = []
    for table in tables:
        columns = [col["name"] for col in inspector.get_columns(table)]
        schema_metadata.append({"table": table, "columns": columns, "object_type": "table"})

    for view in views:
        try:
            columns = [col["name"] for col in inspector.get_columns(view)]
        except Exception:
            columns = []
        schema_metadata.append({"table": view, "columns": columns, "object_type": "view"})

    relationships = []
    for table in tables:
        for fk in inspector.get_foreign_keys(table):
            for col, ref_col in zip(fk["constrained_columns"], fk["referred_columns"]):
                relationships.append(
                    {
                        "TABLE_NAME": table,
                        "COLUMN_NAME": col,
                        "REFERENCED_TABLE_NAME": fk["referred_table"],
                        "REFERENCED_COLUMN_NAME": ref_col,
                    }
                )

    relationships_df = pd.DataFrame(relationships)
    return schema_metadata, relationships_df


def generate_table_semantics(schema_metadata):
    """Generate a short semantic description for each table using the LLM."""
    table_semantics = []

    for item in schema_metadata:
        table_name = item["table"]
        columns = item["columns"]
        object_type = item.get("object_type", "table")

        prompt = SEMANTIC_PROMPT.format(
            table_name=table_name,
            columns=", ".join(columns),
        )

        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/generate",
                json={
                    "model": MODELS["semantic_generator"],
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0,
                    "num_predict": 40,
                },
                timeout=30,
            )
            semantic_text = response.json()["response"].strip()
        except Exception as e:
            logger.warning("Failed to generate semantics for %s: %s", table_name, e)
            semantic_text = f"{object_type.title()} {table_name} with columns: {', '.join(columns)}"

        table_semantics.append({"table": table_name, "semantic": semantic_text})

    return table_semantics


def extract_business_rules(schema_metadata, engine):
    """Return database-specific business rules.

    The previous implementation assumed a particular schema pattern
    (`from_date`/`to_date`) and a sentinel value. That was too brittle
    for a variable user-supplied database, so we leave this empty unless
    rules are provided from a real metadata source.
    """
    return []


def build_schema_texts(schema_metadata, table_semantics):
    """Build compact text representations of the schema for embedding."""
    schema_texts = []

    semantics_by_table = {item["table"]: item["semantic"] for item in table_semantics}

    for item in schema_metadata:
        table_name = item["table"]
        columns = item["columns"]
        object_type = item.get("object_type", "table")
        semantic_description = semantics_by_table.get(table_name, "")

        text = f"""
Object Type: {object_type}
Name: {table_name}

Meaning:
{semantic_description}

Columns:
{chr(10).join('- ' + c for c in columns)}
"""
        schema_texts.append(text)

    return schema_texts


def build_faiss_index(schema_texts, embedding_model):
    """Generate schema embeddings and build a FAISS index."""
    logger.info("Generating schema embeddings...")
    schema_embeddings = embedding_model.encode(schema_texts).astype("float32")

    dimension = schema_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(schema_embeddings)

    logger.info("FAISS index built with %s entries", len(schema_texts))
    return index, schema_embeddings


def build_relationship_text(relationships_df):
    """Format relationships as plain text for LLM context."""
    if relationships_df.empty:
        return ""

    relationship_text = ""
    for _, row in relationships_df.iterrows():
        relationship_text += f"""
{row['TABLE_NAME']}.{row['COLUMN_NAME']}
references
{row['REFERENCED_TABLE_NAME']}.{row['REFERENCED_COLUMN_NAME']}
"""
    return relationship_text


def build_schema_signature(schema_metadata, relationships_df):
    """Create a stable fingerprint for the current database structure."""
    payload = {
        "schema_metadata": schema_metadata,
        "relationships": relationships_df.to_dict("records") if not relationships_df.empty else [],
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def save_cache(data_dict):
    """Save initialization results to cache files."""
    for key, filepath in CACHE_FILES.items():
        if key not in data_dict:
            continue

        value = data_dict[key]

        try:
            if isinstance(value, faiss.Index):
                faiss.write_index(value, str(filepath))
            elif isinstance(value, np.ndarray):
                np.save(filepath, value)
            elif isinstance(value, pd.DataFrame):
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(value.to_dict("records"), f, indent=2, default=str)
            else:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(value, f, indent=2, default=str)
            logger.info("Saved cache: %s", filepath)
        except Exception as e:
            logger.warning("Failed to save cache %s: %s", filepath, e)


def load_cache():
    """Load cached initialization data if available."""
    cache = {}

    for key, filepath in CACHE_FILES.items():
        if not filepath.exists():
            continue

        try:
            if filepath.suffix == ".npy":
                cache[key] = np.load(filepath, allow_pickle=True)
            elif filepath.suffix == ".index":
                cache[key] = faiss.read_index(str(filepath))
            else:
                with open(filepath, "r", encoding="utf-8") as f:
                    cache[key] = json.load(f)
            logger.info("Loaded cache: %s", filepath)
        except Exception as e:
            logger.warning("Failed to load cache %s: %s", filepath, e)

    return cache


def _rebuild_index_from_embeddings(schema_embeddings):
    """Rebuild a FAISS index when the cached index is missing."""
    embeddings = np.asarray(schema_embeddings, dtype="float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def initialize(database_url=None, uploaded_files=None):
    """
    Main initialization function.
    Returns all components needed for the pipeline.
    """
    logger.info("Starting AI Database Assistant initialization...")

    engine, source_info = connect_database(
        database_url=database_url,
        uploaded_files=uploaded_files,
    )
    embedding_model = load_embedding_model()
    schema_metadata, relationships_df = extract_schema_metadata(engine)
    current_signature = build_schema_signature(schema_metadata, relationships_df)
    cached = load_cache()

    has_core_cache = all(
        key in cached
        for key in ["schema_metadata", "table_semantics", "schema_texts", "business_rules"]
    )
    cache_is_valid = (
        has_core_cache
        and cached.get("schema_metadata") == schema_metadata
        and cached.get("business_rules", []) == []
    )
    cached_relationships = cached.get("relationships")
    if cached_relationships is not None:
        cache_is_valid = cache_is_valid and (
            cached_relationships == relationships_df.to_dict("records")
        )
    if not cache_is_valid and cached.get("schema_signature") == current_signature:
        cache_is_valid = True

    if cache_is_valid and has_core_cache:
        logger.info("Using cached initialization data")

        table_semantics = cached["table_semantics"]
        business_rules = []
        schema_texts = cached["schema_texts"]
        index = cached.get("index")
        schema_embeddings = cached.get("schema_embeddings")

        if index is None:
            if schema_embeddings is not None:
                index = _rebuild_index_from_embeddings(schema_embeddings)
            else:
                index, schema_embeddings = build_faiss_index(schema_texts, embedding_model)

        relationship_text = build_relationship_text(relationships_df)

        return {
            "engine": engine,
            "sql_dialect": engine.dialect.name,
            "source_info": source_info,
            "embedding_model": embedding_model,
            "schema_metadata": schema_metadata,
            "relationships_df": relationships_df,
            "table_semantics": table_semantics,
            "business_rules": business_rules,
            "schema_texts": schema_texts,
            "index": index,
            "relationship_text": relationship_text,
            "schema_embeddings": schema_embeddings,
            "schema_signature": current_signature,
        }

    logger.info("Cache miss or schema change detected, rebuilding semantic artifacts")

    table_semantics = generate_table_semantics(schema_metadata)
    business_rules = extract_business_rules(schema_metadata, engine)
    schema_texts = build_schema_texts(schema_metadata, table_semantics)
    index, schema_embeddings = build_faiss_index(schema_texts, embedding_model)
    relationship_text = build_relationship_text(relationships_df)

    cache_data = {
        "schema_signature": current_signature,
        "schema_metadata": schema_metadata,
        "table_semantics": table_semantics,
        "business_rules": business_rules,
        "schema_texts": schema_texts,
        "schema_embeddings": schema_embeddings,
        "index": index,
        "relationships": relationships_df.to_dict("records") if not relationships_df.empty else [],
    }
    save_cache(cache_data)

    logger.info("Initialization complete!")

    return {
        "engine": engine,
        "sql_dialect": engine.dialect.name,
        "source_info": source_info,
        "embedding_model": embedding_model,
        "schema_metadata": schema_metadata,
        "relationships_df": relationships_df,
        "table_semantics": table_semantics,
        "business_rules": business_rules,
        "schema_texts": schema_texts,
        "index": index,
        "relationship_text": relationship_text,
        "schema_embeddings": schema_embeddings,
        "schema_signature": current_signature,
    }
