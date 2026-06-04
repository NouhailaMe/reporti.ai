"""Test SQL generation directly"""
import os

from ai_init import initialize
from pipeline import DatabaseAssistant

# Initialize
print("📦 Initializing...")
database_url = os.getenv("DATABASE_URL")
if not database_url:
    raise SystemExit(
        "Set DATABASE_URL before running this test, or call initialize(database_url=..., uploaded_files=...)."
    )

init_data = initialize(database_url=database_url)
assistant = DatabaseAssistant(init_data)

# Test question
question = "Show all departments"
print(f"\n❓ Question: {question}")

# Manual test
retrieved_indices, retrieved_schemas = assistant._retrieve_relevant_schemas(question)
print(f"\n📊 Retrieved tables: {[assistant.schema_metadata[idx]['table'] for idx in retrieved_indices]}")

normalized = assistant._normalize_question(question, "\n".join(retrieved_schemas))
print(f"✏️ Normalized: {normalized}")

expanded_indices = assistant._expand_schema_with_relationships(
    [int(idx) for idx in retrieved_indices], 
    question_text=normalized
)
print(f"🔗 Expanded tables: {[assistant.schema_metadata[idx]['table'] for idx in expanded_indices]}")

mini_schema = assistant._build_mini_schema(expanded_indices)
semantic_context = assistant._get_semantic_context(expanded_indices)

print(f"\n📋 Mini Schema (first 300 chars):\n{mini_schema[:300]}")

# Generate SQL
sql = assistant._generate_sql(
    normalized, mini_schema, semantic_context,
    assistant.relationship_text, assistant.business_rules
)

print(f"\n🤖 Generated SQL:\n{sql}")
