import requests
import pandas as pd
import numpy as np
import faiss

from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded")

DATABASE_URL = (
    "mysql+pymysql://root:@localhost:3306/employees"
)

engine = create_engine(
    DATABASE_URL
)

print("MySQL connected")

tables_query = """
SELECT table_name
FROM information_schema.tables
WHERE table_schema = DATABASE();
"""

tables_df = pd.read_sql(
    tables_query,
    engine
)

print("Tables loaded")

schema_metadata = []

for table in tables_df["table_name"]:

    columns_query = f"""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = DATABASE()
    AND table_name = '{table}';
    """

    columns_df = pd.read_sql(
        columns_query,
        engine
    )

    schema_metadata.append({
        "table": table,
        "columns": columns_df[
            "column_name"
        ].tolist()
    })

print("Schema metadata loaded")

relationships_query = """
SELECT
    TABLE_NAME,
    COLUMN_NAME,
    REFERENCED_TABLE_NAME,
    REFERENCED_COLUMN_NAME
FROM information_schema.KEY_COLUMN_USAGE
WHERE REFERENCED_TABLE_NAME IS NOT NULL
AND TABLE_SCHEMA = DATABASE();
"""

relationships_df = pd.read_sql(
    relationships_query,
    engine
)

print("Relationships loaded")

table_semantics = []

for item in schema_metadata:

    table_name = item["table"]

    columns = item["columns"]

    semantic_prompt = f"""
You are a database analyst.

Understand the purpose of this SQL table.

Table:
{table_name}

Columns:
{', '.join(columns)}

Explain:
- what this table represents
- what records it stores
- when this table should be used in SQL queries

Keep answer short.
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral:7b",
            "prompt": semantic_prompt,
            "stream": False,
            "temperature": 0,
            "num_predict": 40
        }
    )

    semantic_text = response.json()[
        "response"
    ].strip()

    table_semantics.append({
        "table": table_name,
        "semantic": semantic_text
    })

print("Table semantics generated")

business_rules = []

for item in schema_metadata:

    table_name = item["table"]

    columns = item["columns"]

    if (
        "from_date" in columns
        and "to_date" in columns
    ):

        sample_query = f"""
        SELECT DISTINCT to_date
        FROM {table_name}
        ORDER BY to_date DESC
        LIMIT 5;
        """

        try:

            sample_df = pd.read_sql(
                sample_query,
                engine
            )

            values = sample_df[
                "to_date"
            ].astype(str).tolist()

            if "9999-01-01" in values:

                rule = f"""
In table {table_name},
to_date = '9999-01-01'
means current active record.
"""

                business_rules.append(
                    rule
                )

        except:
            pass

print("Business rules generated")

schema_texts = []

for item in schema_metadata:

    table_name = item["table"]

    columns = item["columns"]

    semantic_description = ""

    for semantic in table_semantics:

        if semantic["table"] == table_name:

            semantic_description = semantic[
                "semantic"
            ]

    text = f"""
Table: {table_name}

Meaning:
{semantic_description}

Columns:
{chr(10).join('- ' + c for c in columns)}
"""

    schema_texts.append(text)

print("Schema texts generated")

schema_embeddings = embedding_model.encode(
    schema_texts
).astype("float32")

print("Embeddings generated")

dimension = schema_embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(
    schema_embeddings
)

print("FAISS ready")

relationship_text = ""

for _, row in relationships_df.iterrows():

    relationship_text += f"""
{row['TABLE_NAME']}.{row['COLUMN_NAME']}
references
{row['REFERENCED_TABLE_NAME']}.{row['REFERENCED_COLUMN_NAME']}
"""

print("Initialization completed")

