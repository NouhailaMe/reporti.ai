import requests
import pandas as pd
import numpy as np
import re
import time
import sqlglot

from ai_init import *

def run_pipeline(user_question):

    # =========================
    # ROUTER
    # =========================

    router_prompt = f"""
You are a request classifier.

Classify the user message.

Return ONLY one word:

DATABASE

or

GENERAL

Message:
{user_question}

Classification:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi3:mini",
            "prompt": router_prompt,
            "stream": False,
            "temperature": 0,
            "num_predict": 5
        }
    )

    intent = (
        response.json()["response"]
        .strip()
        .upper()
    )

    # =========================
    # GENERAL CHAT
    # =========================

    if "GENERAL" in intent:

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral:7b",
                "prompt": user_question,
                "stream": False,
                "temperature": 0.7,
                "num_predict": 200
            }
        )

        answer = (
            response.json()["response"]
            .strip()
        )

        return {
            "question": user_question,
            "intent": "GENERAL",
            "answer": answer,
            "rows": []
        }

    # =========================
    # DATABASE PIPELINE
    # =========================

    query_embedding = embedding_model.encode(
        [user_question]
    ).astype("float32")

    # =========================
    # PRE-NORMALIZATION RETRIEVAL
    # =========================

    query_embedding = embedding_model.encode(
        [user_question]
    ).astype("float32")

    D, I = index.search(
        query_embedding,
        k=3
    )

    pre_normalization_schema = ""

    for idx in I[0]:

        pre_normalization_schema += (
            schema_texts[idx]
        )

        pre_normalization_schema += "\n"

    # =========================
    # NORMALIZATION
    # =========================

    intent_prompt = f"""
You are a semantic database request clarifier.

DATABASE CONTEXT:
{pre_normalization_schema}

DATABASE RELATIONSHIPS:
{relationship_text}

User Request:
{user_question}

Clarified Request:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral:7b",
            "prompt": intent_prompt,
            "stream": False,
            "temperature": 0,
            "num_predict": 30
        }
    )

    normalized_question = (
        response.json()["response"]
        .strip()
    )

    # =========================
    # SEMANTIC RETRIEVAL
    # =========================

    query_embedding = embedding_model.encode(
        [normalized_question]
    ).astype("float32")

    D, I = index.search(
        query_embedding,
        k=3
    )

    relevant_semantics = []

    for idx in I[0]:

        relevant_semantics.append(
            table_semantics[idx]["semantic"]
        )

    semantic_context = "\n".join(
        relevant_semantics
    )

    expanded_indices = set(I[0])

    retrieved_tables = []

    for idx in I[0]:

        table_name = schema_metadata[idx][
            "table"
        ]

        retrieved_tables.append(
            table_name
        )

    for _, row in relationships_df.iterrows():

        source_table = row[
            "TABLE_NAME"
        ]

        target_table = row[
            "REFERENCED_TABLE_NAME"
        ]

        if source_table in retrieved_tables:

            for i, item in enumerate(
                schema_metadata
            ):

                if (
                    item["table"]
                    == target_table
                ):

                    expanded_indices.add(
                        i
                    )

        if target_table in retrieved_tables:

            for i, item in enumerate(
                schema_metadata
            ):

                if (
                    item["table"]
                    == source_table
                ):

                    expanded_indices.add(
                        i
                    )

    mini_schema = ""

    for idx in expanded_indices:

        mini_schema += (
            schema_texts[idx]
        )

        mini_schema += "\n"

    # =========================
    # SQL GENERATION
    # =========================

    sql_prompt = f"""
You are an expert MySQL SQL generator.

DATABASE SCHEMA:
{mini_schema}

DATABASE RELATIONSHIPS:
{relationship_text}

DATABASE SEMANTICS:
{semantic_context}

DATABASE BUSINESS RULES:
{chr(10).join(business_rules)}

Question:
{normalized_question}

SQL:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model":
            "deepseek-coder:6.7b",
            "prompt":
            sql_prompt,
            "stream":
            False,
            "temperature":
            0,
            "num_predict":
            150
        }
    )

    sql_query = (
        response.json()["response"]
    )

    # =========================
    # CLEAN SQL
    # =========================

    sql_query = re.sub(
        r"```sql",
        "",
        sql_query
    )

    sql_query = re.sub(
        r"```",
        "",
        sql_query
    )

    sql_query = re.sub(
        r"<\|.*?\|>",
        "",
        sql_query
    )

    sql_query = re.sub(
        r"▁",
        "",
        sql_query
    )

    sql_query = re.sub(
        r"\s+",
        " ",
        sql_query
    )

    sql_query = sql_query.strip()

    sql_query = (
        sql_query.split(";")[0]
        + ";"
    )

    # =========================
    # REPAIR
    # =========================

    try:

        pd.read_sql(
            f"EXPLAIN {sql_query}",
            engine
        )

        repaired_sql = sql_query

    except Exception as e:

        repair_prompt = f"""
You are an expert MySQL SQL fixer.

DATABASE SCHEMA:
{mini_schema}

DATABASE RELATIONSHIPS:
{relationship_text}

ORIGINAL QUESTION:
{normalized_question}

INVALID SQL:
{sql_query}

DATABASE ERROR:
{str(e)}

CORRECTED SQL:
"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model":
                "deepseek-coder:6.7b",
                "prompt":
                repair_prompt,
                "stream":
                False,
                "temperature":
                0,
                "num_predict":
                150
            }
        )

        repaired_sql = (
            response.json()["response"]
            .strip()
        )

        repaired_sql = (
            repaired_sql
            .split(";")[0]
            + ";"
        )

    # =========================
    # EXECUTION
    # =========================

    try:

        df = pd.read_sql(
            repaired_sql,
            engine
        )

    except Exception as e:

        return {
            "question":
            user_question,

            "normalized_question":
            normalized_question,

            "sql":
            repaired_sql,

            "error":
            str(e)
        }

    # =========================
    # HUMANIZATION
    # =========================

    if len(df) == 0:

        result_preview = (
            "No rows returned"
        )

    else:

        result_preview = (
            df.head(10)
            .to_string(index=False)
        )

    human_prompt = f"""
You are a business data analyst.

USER QUESTION:
{user_question}

SQL RESULT:
{result_preview}

ANSWER:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model":
            "mistral:7b",
            "prompt":
            human_prompt,
            "stream":
            False,
            "temperature":
            0.2,
            "num_predict":
            120
        }
    )

    final_answer = (
        response.json()["response"]
        .strip()
    )

    return {

        "question":
        user_question,

        "normalized_question":
        normalized_question,

        "sql":
        repaired_sql,

        "answer":
        final_answer,

        "rows":
        df.to_dict(
            orient="records"
        )
    }
