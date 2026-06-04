"""
Main pipeline module - handles question processing end-to-end
Follows EXACT workflow from notebook - NO model changes
"""
import re
import time
import json
import logging
import pandas as pd
import numpy as np
import faiss
import requests
import sqlglot
from sqlalchemy import inspect, text
from numpy.linalg import norm

from config import (
    MODELS, OLLAMA_BASE_URL, RAG_TOP_K, DEFAULT_LIMIT,
    SQL_MAX_TOKENS, SQL_TEMPERATURE, CACHE_FILES, OLLAMA_KEEP_ALIVE,
    QUESTION_CACHE_VERSION
)
from prompts import (
    PLANNER_PROMPT, NORMALIZER_PROMPT, SQL_GENERATION_PROMPT,
    SQL_REPAIR_PROMPT, HUMANIZE_PROMPT, GENERAL_PROMPT, METADATA_PROMPT
)

logger = logging.getLogger(__name__)


class DatabaseAssistant:
    """Main pipeline class for natural language to SQL conversion"""
    
    def __init__(self, init_data=None):
        """Initialize with pre-loaded components from ai_init"""
        if init_data is None:
            from ai_init import initialize
            init_data = initialize()
        
        self.engine = init_data["engine"]
        self.sql_dialect = init_data.get("sql_dialect", getattr(self.engine.dialect, "name", "sql"))
        self.embedding_model = init_data["embedding_model"]
        self.schema_metadata = init_data["schema_metadata"]
        self.relationships_df = init_data["relationships_df"]
        self.table_semantics = init_data["table_semantics"]
        self.business_rules = init_data["business_rules"]
        self.schema_texts = init_data["schema_texts"]
        self.index = init_data["index"]
        self.relationship_text = init_data["relationship_text"]
        self.schema_signature = init_data.get("schema_signature", "")
        self.inspector = inspect(self.engine)
        self.table_names = self.inspector.get_table_names()
        self.view_names = self.inspector.get_view_names()
        self.schema_object_names = self.table_names + self.view_names
        self.question_cache = self._load_question_cache()
        self.intent_router = self._build_intent_router()
        
        # Build lookup maps
        self._table_to_idx = {
            item["table"]: i for i, item in enumerate(self.schema_metadata)
        }

        self._warm_models()

    def _build_intent_router(self):
        examples = [
            {"intent": "GENERAL", "metadata_action": "unknown", "text": "hi"},
            {"intent": "GENERAL", "metadata_action": "unknown", "text": "hello"},
            {"intent": "GENERAL", "metadata_action": "unknown", "text": "how are you"},
            {"intent": "GENERAL", "metadata_action": "unknown", "text": "thanks"},
            {"intent": "GENERAL", "metadata_action": "unknown", "text": "explain joins"},
            {"intent": "GENERAL", "metadata_action": "unknown", "text": "help me understand this"},
            {"intent": "METADATA", "metadata_action": "object_count", "text": "how many tables are in the database"},
            {"intent": "METADATA", "metadata_action": "object_count", "text": "how many views are there"},
            {"intent": "METADATA", "metadata_action": "list_objects", "text": "list all tables and views"},
            {"intent": "METADATA", "metadata_action": "column_list", "text": "what columns does this table have"},
            {"intent": "METADATA", "metadata_action": "record_count", "text": "how many records are there"},
            {"intent": "METADATA", "metadata_action": "record_count", "text": "count all rows"},
            {"intent": "DATABASE", "metadata_action": "unknown", "text": "show the latest announcements"},
            {"intent": "DATABASE", "metadata_action": "unknown", "text": "find the newest records"},
            {"intent": "DATABASE", "metadata_action": "unknown", "text": "what is the average amount"},
        ]
        texts = [item["text"] for item in examples]
        vectors = self.embedding_model.encode(texts).astype("float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
        for item, vector in zip(examples, vectors):
            item["vector"] = vector
        return examples

    def _semantic_route(self, question):
        query_vec = self.embedding_model.encode([question]).astype("float32")[0]
        query_norm = norm(query_vec)
        if query_norm == 0:
            return self._default_plan()
        query_vec = query_vec / query_norm

        best = None
        best_score = -1.0
        for example in self.intent_router:
            score = float(np.dot(query_vec, example["vector"]))
            if score > best_score:
                best_score = score
                best = example

        plan = self._default_plan()
        if best is not None:
            plan.update({
                "intent": best["intent"],
                "metadata_action": best.get("metadata_action", "unknown"),
                "confidence": best_score,
            })
        return plan
    
    def _call_ollama(self, model, prompt, max_tokens=150, temperature=0):
        """Helper to call Ollama API"""
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "keep_alive": OLLAMA_KEEP_ALIVE,
                },
                timeout=300
            )
            return response.json()["response"].strip()
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return ""
    
    def _clean_sql(self, sql_text):
        """Clean SQL output from LLM artifacts"""
        sql_text = re.sub(r"```sql\s*|\s*```", "", sql_text, flags=re.IGNORECASE)
        sql_text = re.sub(r"<\|.*?\|>", "", sql_text)
        sql_text = re.sub(r"▁", "", sql_text)
        sql_text = re.sub(r"\s+", " ", sql_text)
        sql_text = sql_text.strip()
        if sql_text and not sql_text.endswith(";"):
            sql_text += ";"
        return sql_text.split(";")[0] + ";"

    def _sqlglot_dialect(self):
        """Map the live database dialect to a sqlglot parser dialect when possible."""
        dialect = (self.sql_dialect or "").lower()
        mapping = {
            "mysql": "mysql",
            "mariadb": "mysql",
            "postgresql": "postgres",
            "postgres": "postgres",
            "sqlite": "sqlite",
            "mssql": "tsql",
            "sqlserver": "tsql",
            "oracle": "oracle",
            "duckdb": "duckdb",
            "bigquery": "bigquery",
            "snowflake": "snowflake",
        }
        return mapping.get(dialect)
    
    def _schema_summary_for_router(self, max_objects=10):
        lines = []
        for item in self.schema_metadata[:max_objects]:
            columns = ", ".join(item.get("columns", [])[:8]) or "no readable columns"
            lines.append(f"{item.get('table', 'unknown')}: {columns}")
        if len(self.schema_metadata) > max_objects:
            lines.append(f"... and {len(self.schema_metadata) - max_objects} more object(s)")
        return "\n".join(lines)

    def _safe_json_extract(self, text):
        if not text:
            return {}
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return {}

    def _default_plan(self):
        return {
            "intent": "GENERAL",
            "metadata_action": "unknown",
            "target_object": None,
            "needs_clarification": False,
            "clarification_question": None,
            "confidence": 0.0,
        }

    def _build_plan_cache_key(self, question):
        normalized = re.sub(r"\s+", " ", (question or "").strip().lower())
        return f"{self.schema_signature}::PLAN::{normalized}"

    def _get_cached_plan(self, question):
        key = self._build_plan_cache_key(question)
        entry = self.question_cache.get("entries", {}).get(key)
        if isinstance(entry, dict):
            return entry
        return None

    def _store_cached_plan(self, question, plan):
        key = self._build_plan_cache_key(question)
        self.question_cache.setdefault("schema_signature", self.schema_signature)
        self.question_cache.setdefault("entries", {})
        self.question_cache["entries"][key] = self._json_safe(plan)
        self._save_question_cache()

    def _build_router_context(self, question):
        retrieved_indices, retrieved_schemas = self._build_metadata_context(question)
        schema_context = "\n".join(retrieved_schemas) if retrieved_schemas else ""
        return schema_context

    def _plan_question(self, question):
        cached = self._get_cached_plan(question)
        if cached is not None:
            return cached

        plan = self._semantic_route(question)
        intent = str(plan.get("intent", "GENERAL")).upper().strip()
        confidence = float(plan.get("confidence", 0.0) or 0.0)

        if intent == "GENERAL" and confidence >= 0.80:
            plan["fast_response"] = True
        elif intent == "METADATA" and confidence >= 0.75:
            plan["fast_response"] = True
            if plan.get("metadata_action") in {"column_list", "record_count"}:
                retrieved_indices, _ = self._build_metadata_context(question)
                if len(retrieved_indices) > 0:
                    guessed = self.schema_metadata[int(retrieved_indices[0])]
                    if guessed:
                        plan["target_object"] = guessed.get("table")
        elif intent == "DATABASE" and confidence >= 0.80:
            plan["fast_response"] = False

        if confidence < 0.60:
            schema_context = self._build_router_context(question)
            prompt = PLANNER_PROMPT.format(
                schema_summary=self._build_schema_summary(),
                schema_context=schema_context or "No specific schema context retrieved.",
                question=question,
            )
            response = self._call_ollama(
                MODELS["router"], prompt, max_tokens=220, temperature=0
            )
            parsed = self._safe_json_extract(response)
            if isinstance(parsed, dict):
                plan.update(parsed)
            intent = str(plan.get("intent", "GENERAL")).upper().strip()

        if intent not in {"GENERAL", "METADATA", "DATABASE"}:
            intent = "GENERAL"
        plan["intent"] = intent
        plan["metadata_action"] = str(plan.get("metadata_action", "unknown")).lower().strip()
        plan["target_object"] = plan.get("target_object") or None
        plan["needs_clarification"] = bool(plan.get("needs_clarification", False))
        plan["clarification_question"] = plan.get("clarification_question") or None

        self._store_cached_plan(question, plan)
        return plan
    
    def _retrieve_relevant_schemas(self, question, k=RAG_TOP_K):
        """Use FAISS to retrieve top-k relevant schema descriptions"""
        query_embedding = self.embedding_model.encode(
            [question]
        ).astype("float32")
        
        D, I = self.index.search(query_embedding, k=min(k, len(self.schema_texts)))
        
        return I[0], [self.schema_texts[idx] for idx in I[0]]
    
    def _expand_schema_with_relationships(self, retrieved_indices, question_text=None):
        """Expand retrieved tables using foreign key relationships"""
        # Convert numpy indices to Python ints
        retrieved_indices = [int(idx) for idx in retrieved_indices]
        
        expanded = set(retrieved_indices)
        retrieved_tables = [
            self.schema_metadata[idx]["table"] for idx in retrieved_indices
        ]
        
        for _, row in self.relationships_df.iterrows():
            source = row["TABLE_NAME"]
            target = row["REFERENCED_TABLE_NAME"]
            
            if source in retrieved_tables and target in self._table_to_idx:
                expanded.add(self._table_to_idx[target])
            if target in retrieved_tables and source in self._table_to_idx:
                expanded.add(self._table_to_idx[source])
        
        return expanded
    
    def _build_mini_schema(self, indices):
        """Build condensed schema text from selected tables"""
        return "\n".join(self.schema_texts[idx] for idx in sorted(indices))
    
    def _normalize_question(self, question, schema_context):
        """Clarify question while preserving intent - EXACT notebook logic"""
        prompt = NORMALIZER_PROMPT.format(
            question=question,
            schema_context=schema_context,
            relationships=self.relationship_text
        )
        return self._call_ollama(
            MODELS["normalizer"], prompt, max_tokens=30, temperature=0
        ) or question
    
    def _generate_sql(self, question, mini_schema, semantics, relationships, rules):
        """Generate SQL query using deepseek-coder:6.7b - EXACT notebook logic"""
        prompt = SQL_GENERATION_PROMPT.format(
            dialect=self.sql_dialect,
            schema=mini_schema,
            relationships=relationships,
            semantics=semantics,
            business_rules="\n".join(rules),
            question=question,
            limit=DEFAULT_LIMIT
        )
        raw_sql = self._call_ollama(
            MODELS["sql_generator"], prompt, 
            max_tokens=min(SQL_MAX_TOKENS, 120), temperature=SQL_TEMPERATURE
        )
        return self._clean_sql(raw_sql)
    
    def _validate_sql(self, sql_query):
        """Validate SQL syntax using sqlglot"""
        try:
            dialect = self._sqlglot_dialect()
            if dialect:
                sqlglot.parse_one(sql_query, read=dialect)
            else:
                sqlglot.parse_one(sql_query)
            return True, None
        except Exception as e:
            return False, str(e)

    def _repair_sql(self, question, invalid_sql, error, mini_schema):
        """Attempt to repair invalid SQL using deepseek-coder:6.7b"""
        prompt = SQL_REPAIR_PROMPT.format(
            dialect=self.sql_dialect,
            schema=mini_schema,
            relationships=self.relationship_text,
            question=question,
            invalid_sql=invalid_sql,
            error=error
        )
        raw_sql = self._call_ollama(
            MODELS["sql_repairer"], prompt,
            max_tokens=min(SQL_MAX_TOKENS, 120), temperature=0
        )
        return self._clean_sql(raw_sql)
    
    def _execute_sql(self, sql_query):
        """Execute SQL and return DataFrame"""
        try:
            df = pd.read_sql(text(sql_query), self.engine)
            return True, df, None
        except Exception as e:
            return False, None, str(e)
    
    def _humanize_answer(self, question, result_preview):
        """Convert SQL results to natural language using mistral:7b"""
        prompt = HUMANIZE_PROMPT.format(
            question=question,
            result_preview=result_preview
        )
        return self._call_ollama(
            MODELS["humanizer"], prompt, max_tokens=120, temperature=0.2
        )

    def _format_simple_result(self, df):
        """Generate a fast direct answer for small result sets without another LLM call."""
        if df is None or df.empty:
            return "No rows were returned."

        if len(df.columns) == 1 and len(df) == 1:
            column = str(df.columns[0])
            value = df.iloc[0, 0]
            return f"{column}: {value}."

        if len(df) == 1 and len(df.columns) <= 3:
            parts = [f"{col}: {df.iloc[0][col]}" for col in df.columns]
            return "; ".join(parts) + "."

        if len(df) <= 3 and len(df.columns) <= 3:
            lines = []
            for _, row in df.iterrows():
                pairs = ", ".join(f"{col}: {row[col]}" for col in df.columns)
                lines.append(pairs)
            return " | ".join(lines) + "."

        return None

    def _quick_general_reply(self, question, plan=None):
        """Fast non-LLM reply for high-confidence small-talk and greetings."""
        plan = plan or self._semantic_route(question)
        if plan.get("intent") != "GENERAL" or float(plan.get("confidence", 0.0) or 0.0) < 0.86:
            return None

        if len((question or "").split()) <= 4:
            return "Hello! I’m ready to help with your database or any other question."

        return None
    
    def _get_semantic_context(self, indices):
        """Get semantic descriptions for retrieved tables"""
        return "\n".join(
            self.table_semantics[idx]["semantic"] for idx in sorted(indices)
            if idx < len(self.table_semantics)
        )

    def _question_cache_path(self):
        return CACHE_FILES["question_cache"]

    def _load_question_cache(self):
        path = self._question_cache_path()
        if not path.exists():
            return {
                "schema_signature": self.schema_signature,
                "version": QUESTION_CACHE_VERSION,
                "entries": {},
            }

        try:
            with open(path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("schema_signature") != self.schema_signature:
                return {
                    "schema_signature": self.schema_signature,
                    "version": QUESTION_CACHE_VERSION,
                    "entries": {},
                }
            if cached.get("version") != QUESTION_CACHE_VERSION:
                return {
                    "schema_signature": self.schema_signature,
                    "version": QUESTION_CACHE_VERSION,
                    "entries": {},
                }
            if not isinstance(cached.get("entries"), dict):
                return {
                    "schema_signature": self.schema_signature,
                    "version": QUESTION_CACHE_VERSION,
                    "entries": {},
                }
            logger.info("Loaded question cache: %s", path)
            return cached
        except Exception as e:
            logger.warning("Failed to load question cache %s: %s", path, e)
            return {
                "schema_signature": self.schema_signature,
                "version": QUESTION_CACHE_VERSION,
                "entries": {},
            }

    def _save_question_cache(self):
        path = self._question_cache_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.question_cache, f, indent=2, ensure_ascii=False)
            logger.info("Saved question cache: %s", path)
        except Exception as e:
            logger.warning("Failed to save question cache %s: %s", path, e)

    def _question_cache_key(self, question, intent):
        normalized = re.sub(r"\s+", " ", (question or "").strip().lower())
        return f"{self.schema_signature}::{intent}::{normalized}"

    def _get_cached_result(self, question, intent):
        key = self._question_cache_key(question, intent)
        entry = self.question_cache.get("entries", {}).get(key)
        if entry is None:
            return None
        return json.loads(json.dumps(entry))

    def _store_cached_result(self, question, intent, result):
        key = self._question_cache_key(question, intent)
        self.question_cache.setdefault("schema_signature", self.schema_signature)
        self.question_cache.setdefault("version", QUESTION_CACHE_VERSION)
        self.question_cache.setdefault("entries", {})
        self.question_cache["entries"][key] = self._json_safe(result)
        self._save_question_cache()

    def _json_safe(self, value):
        if isinstance(value, dict):
            return {key: self._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [self._json_safe(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _parse_intent(self, response):
        text = (response or "").upper()
        for label in ("GENERAL", "METADATA", "DATABASE"):
            if re.search(rf"\b{label}\b", text):
                return label
        return "GENERAL"

    def _build_schema_summary(self, max_objects=20):
        preview_lines = []
        for item in self.schema_metadata[:max_objects]:
            object_type = item.get("object_type", "table")
            columns = ", ".join(item.get("columns", [])) or "no readable columns"
            preview_lines.append(f"- {item.get('table', 'unknown')} ({object_type}): {columns}")

        remaining = len(self.schema_metadata) - len(preview_lines)
        if remaining > 0:
            preview_lines.append(f"- ... and {remaining} more schema object(s)")

        tables = ", ".join(self.table_names) if self.table_names else "none"
        views = ", ".join(self.view_names) if self.view_names else "none"
        return (
            f"Tables detected ({len(self.table_names)}): {tables}\n"
            f"Views detected ({len(self.view_names)}): {views}\n"
            f"Schema preview:\n" + "\n".join(preview_lines)
        )

    def _warm_models(self):
        """Prime the Ollama models once per assistant so the first chat is fast."""
        warmups = [
            (MODELS["router"], "Question: hi\nIntent:", 5, 0),
            (MODELS["general"], "Say hello in one short sentence.", 20, 0.2),
            (MODELS["sql_generator"], "SELECT 1;", 10, 0),
        ]
        for model, prompt, max_tokens, temperature in warmups:
            self._call_ollama(model, prompt, max_tokens=max_tokens, temperature=temperature)

    def _build_metadata_context(self, question):
        retrieved_indices, retrieved_schemas = self._retrieve_relevant_schemas(
            question,
            k=min(RAG_TOP_K, max(1, len(self.schema_texts)))
        )
        if isinstance(retrieved_schemas, list):
            retrieved_context = "\n".join(retrieved_schemas)
        else:
            retrieved_context = ""
        return retrieved_indices, retrieved_context

    def _resolve_target_object(self, target_object):
        if not target_object:
            return None
        target = str(target_object).strip().lower()
        for item in self.schema_metadata:
            name = str(item.get("table", "")).strip().lower()
            if name == target:
                return item
        for item in self.schema_metadata:
            name = str(item.get("table", "")).strip().lower()
            if target in name or name in target:
                return item
        return None

    def _count_rows_for_table(self, table_name):
        try:
            count_sql = text(f"SELECT COUNT(*) AS row_count FROM `{table_name}`")
            df = pd.read_sql(count_sql, self.engine)
            if not df.empty:
                return int(df.iloc[0]["row_count"])
        except Exception as e:
            logger.warning("Failed to count rows for %s: %s", table_name, e)
        return None

    def _mysql_approx_total_records(self):
        try:
            database_name = getattr(self.engine.url, "database", None)
            if not database_name:
                return None, []
            query = text(
                """
                SELECT TABLE_NAME, COALESCE(TABLE_ROWS, 0) AS ROWS_ESTIMATE
                FROM information_schema.tables
                WHERE table_schema = :database_name
                  AND table_type = 'BASE TABLE'
                ORDER BY TABLE_NAME
                """
            )
            df = pd.read_sql(query, self.engine, params={"database_name": database_name})
            if df.empty:
                return 0, []
            total = int(df["ROWS_ESTIMATE"].fillna(0).astype("int64").sum())
            details = [
                {"table": row["TABLE_NAME"], "rows": int(row["ROWS_ESTIMATE"] or 0)}
                for _, row in df.iterrows()
            ]
            return total, details
        except Exception as e:
            logger.warning("Failed to fetch MySQL record estimates: %s", e)
            return None, []

    def _total_record_summary(self):
        dialect = (self.sql_dialect or "").lower()
        if dialect in {"mysql", "mariadb"}:
            total, details = self._mysql_approx_total_records()
            if total is not None:
                table_count = len(details)
                return {
                    "total": total,
                    "table_count": table_count,
                    "details": details,
                    "mode": "approximate" if details else "exact",
                }

        details = []
        total = 0
        for item in self.schema_metadata:
            table_name = item.get("table")
            if not table_name:
                continue
            count = self._count_rows_for_table(table_name)
            if count is None:
                continue
            details.append({"table": table_name, "rows": count})
            total += count
        return {
            "total": total,
            "table_count": len(details),
            "details": details,
            "mode": "exact" if details else "unknown",
        }

    def _metadata_answer_from_plan(self, question, plan):
        action = (plan or {}).get("metadata_action", "unknown")
        target_object = self._resolve_target_object((plan or {}).get("target_object"))

        if action == "record_count":
            if target_object:
                count = self._count_rows_for_table(target_object["table"])
                if count is not None:
                    return (
                        f"{target_object['table']} currently has {count} record"
                        f"{'' if count == 1 else 's'}."
                    )
            summary = self._total_record_summary()
            if summary["mode"] == "approximate":
                return (
                    f"The database has approximately {summary['total']} records "
                    f"across {summary['table_count']} tables."
                )
            if summary["mode"] == "exact":
                return (
                    f"The database has {summary['total']} records across "
                    f"{summary['table_count']} tables."
                )

        if action == "object_count":
            return (
                f"The database currently has {len(self.table_names)} tables and "
                f"{len(self.view_names)} views."
            )

        if action == "list_objects":
            table_list = ", ".join(self.table_names) if self.table_names else "no tables"
            view_list = ", ".join(self.view_names) if self.view_names else "no views"
            return f"Tables: {table_list}. Views: {view_list}."

        if action == "column_list":
            if target_object:
                cols = ", ".join(target_object.get("columns", [])) or "no readable columns"
                return f"{target_object['table']} has these columns: {cols}."

        return None

    def process_metadata_question(self, question, plan=None):
        """Answer schema/structure questions using the live schema plus vector retrieval."""
        cached = self._get_cached_result(question, "METADATA")
        if cached is not None:
            cached["cached"] = True
            return cached

        if plan and plan.get("fast_response"):
            direct_answer = self._metadata_answer_from_plan(question, plan or {})
            if direct_answer:
                result = {
                    "intent": "METADATA",
                    "question": question,
                    "normalized_question": question,
                    "sql": None,
                    "answer": direct_answer,
                    "rows": [],
                    "error": None,
                }
                self._store_cached_result(question, "METADATA", result)
                return result

        direct_answer = self._metadata_answer_from_plan(question, plan or {})
        if direct_answer:
            result = {
                "intent": "METADATA",
                "question": question,
                "normalized_question": question,
                "sql": None,
                "answer": direct_answer,
                "rows": [],
                "error": None,
            }
            self._store_cached_result(question, "METADATA", result)
            return result

        _, schema_context = self._build_metadata_context(question)
        prompt = METADATA_PROMPT.format(
            schema_summary=self._build_schema_summary(),
            schema_context=schema_context or "No specific schema context retrieved.",
            relationships=self.relationship_text,
            question=question,
        )
        answer = self._call_ollama(
            MODELS["general"],
            prompt,
            max_tokens=120,
            temperature=0.1,
        )
        result = {
            "intent": "METADATA",
            "question": question,
            "normalized_question": question,
            "sql": None,
            "answer": answer,
            "rows": [],
            "error": None,
        }
        self._store_cached_result(question, "METADATA", result)
        return result
    
    def process_general_question(self, question, plan=None):
        """Handle non-database questions directly without database access."""
        start_time = time.time()
        cached = self._get_cached_result(question, "GENERAL")
        if cached is not None:
            cached["cached"] = True
            return cached

        quick_reply = self._quick_general_reply(question, plan=plan)
        if quick_reply is not None:
            result = {
                "intent": "GENERAL",
                "question": question,
                "answer": quick_reply,
                "sql": None,
                "rows": [],
                "timings": {"total_seconds": round(time.time() - start_time, 2), "mode": "fast"},
            }
            self._store_cached_result(question, "GENERAL", result)
            return result

        prompt = GENERAL_PROMPT.format(question=question)
        answer = self._call_ollama(
            MODELS["general"],
            prompt,
            max_tokens=80,
            temperature=0.3
        )
        result = {
            "intent": "GENERAL",
            "question": question,
            "answer": answer,
            "sql": None,
            "rows": [],
            "timings": {"total_seconds": round(time.time() - start_time, 2), "mode": "llm"},
        }
        self._store_cached_result(question, "GENERAL", result)
        return result
    
    def process_database_question(self, question, plan=None):
        """Full pipeline for database queries - EXACT notebook workflow"""
        cached = self._get_cached_result(question, "DATABASE")
        if cached is not None:
            cached["cached"] = True
            return cached

        if plan and plan.get("needs_clarification"):
            clarification = plan.get("clarification_question") or (
                "I need one more detail to answer that database question safely."
            )
            return {
                "intent": "DATABASE",
                "question": question,
                "normalized_question": question,
                "sql": None,
                "answer": clarification,
                "rows": [],
                "error": None,
            }

        start_time = time.time()

        result = {
            "intent": "DATABASE",
            "question": question,
            "normalized_question": None,
            "sql": None,
            "answer": None,
            "rows": [],
            "error": None,
            "timings": {},
        }
        timing_mark = time.time()
        
        try:
            # Step 1: Retrieve relevant schemas (pre-normalization for context)
            retrieved_indices, retrieved_schemas = self._retrieve_relevant_schemas(question)
            pre_norm_schema = "\n".join(retrieved_schemas)
            result["timings"]["retrieve_initial"] = round(time.time() - timing_mark, 2)
            
            # Step 2: Normalize question (preserve intent)
            if plan and (plan.get("fast_response") or float(plan.get("confidence", 0.0) or 0.0) >= 0.85):
                normalized = question
            else:
                normalized = self._normalize_question(question, pre_norm_schema)
            result["normalized_question"] = normalized
            result["timings"]["normalize"] = round(time.time() - timing_mark, 2)
            
            # Step 3: Re-retrieve with normalized question
            retrieved_indices, _ = self._retrieve_relevant_schemas(normalized)
            
            # Step 4: Expand schema with relationships (RAG expansion)
            expanded_indices = self._expand_schema_with_relationships(retrieved_indices)
            result["timings"]["rag_expand"] = round(time.time() - timing_mark, 2)

            # Step 5: Build mini schema and semantic context
            mini_schema = self._build_mini_schema(expanded_indices)
            semantic_context = self._get_semantic_context(expanded_indices)
            result["timings"]["schema_build"] = round(time.time() - timing_mark, 2)
            
            # Step 6: Generate SQL with deepseek-coder:6.7b
            sql = self._generate_sql(
                normalized, mini_schema, semantic_context,
                self.relationship_text, self.business_rules
            )
            result["sql"] = sql
            result["timings"]["sql_generate"] = round(time.time() - timing_mark, 2)
            
            # Safety check: empty SQL
            if not sql or sql.strip() == "" or sql.strip() == ";":
                result["error"] = "Failed to generate SQL query"
                result["answer"] = "I couldn't generate a valid SQL query. Please rephrase."
                return result
            
            # Step 7: Validate SQL with sqlglot
            is_valid, error = self._validate_sql(sql)
            if not is_valid:
                logger.warning(f"SQL validation failed: {error}")
                sql = self._repair_sql(normalized, sql, error, mini_schema)
                result["sql"] = sql
            result["timings"]["sql_validate"] = round(time.time() - timing_mark, 2)
            
            # Step 8: Execute SQL
            success, df, exec_error = self._execute_sql(sql)
            if not success:
                logger.error(f"SQL execution failed: {exec_error}")
                # Try repair once more with execution error
                sql = self._repair_sql(normalized, sql, exec_error, mini_schema)
                result["sql"] = sql
                success, df, exec_error = self._execute_sql(sql)
            
            if not success:
                result["error"] = exec_error
                result["answer"] = f"Error executing query: {exec_error}"
                return result
            result["timings"]["sql_execute"] = round(time.time() - timing_mark, 2)
            
            # Step 9: Prepare result preview
            if len(df) == 0:
                result_preview = "No rows returned"
            else:
                result_preview = df.head(10).to_string(index=False)
            
            # Step 10: Humanize answer, but skip another model call when the result is already simple.
            answer = self._format_simple_result(df)
            if answer is None:
                answer = self._humanize_answer(question, result_preview)
            result["answer"] = answer
            result["rows"] = df.to_dict('records')
            result["timings"]["answer"] = round(time.time() - timing_mark, 2)
            
            elapsed = time.time() - start_time
            logger.info(f"Pipeline completed in {elapsed:.2f}s")
            result["timings"]["total_seconds"] = round(elapsed, 2)
            self._store_cached_result(question, "DATABASE", result)
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            result["error"] = str(e)
            result["answer"] = f"An error occurred: {str(e)}"
        
        return result
    
    def run(self, question):
        """Main entry point - routes and processes question"""
        cached = self._get_cached_result(question, "AUTO")
        if cached is not None:
            cached["cached"] = True
            return cached

        plan = self._plan_question(question)
        intent = plan.get("intent", "GENERAL")

        if intent == "GENERAL":
            result = self.process_general_question(question, plan=plan)
        elif intent == "METADATA":
            result = self.process_metadata_question(question, plan=plan)
        else:
            result = self.process_database_question(question, plan=plan)

        self._store_cached_result(question, "AUTO", result)
        return result


# Global instance for convenience
_assistant = None

def run_pipeline(question, init_data=None):
    """Global function to run the pipeline (lazy initialization)."""
    global _assistant
    if init_data is not None:
        assistant = DatabaseAssistant(init_data)
        return assistant.run(question)

    if _assistant is None:
        raise RuntimeError(
            "No database session is loaded. Use initialize(database_url=...) or upload a database through the frontend."
        )
    return _assistant.run(question)
