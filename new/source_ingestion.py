"""
Helpers for turning user-provided sources into a SQLAlchemy engine.

Supported sources:
- Existing database connection string
- Uploaded MySQL SQL dumps
- SQLite database files
- CSV files
- Excel workbooks
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd
import pymysql
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine.url import make_url, URL

from config import MYSQL_SERVER_URL

logger = logging.getLogger(__name__)

SQLITE_EXTENSIONS = {".db", ".sqlite", ".sqlite3"}
CSV_EXTENSIONS = {".csv"}
EXCEL_EXTENSIONS = {".xlsx", ".xls"}
SQL_EXTENSIONS = {".sql"}


def _as_path(item) -> Path:
    path = getattr(item, "name", item)
    return Path(path)


def _sanitize_table_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "table"
    if cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned.lower()


def _unique_name(base: str, used: set[str]) -> str:
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def _is_mysql_url(value: str | None) -> bool:
    if not value:
        return False
    try:
        return make_url(value).get_backend_name() == "mysql"
    except Exception:
        return False


def _mysql_server_url(source_database_url: str | None = None) -> str:
    if _is_mysql_url(source_database_url):
        return source_database_url
    return MYSQL_SERVER_URL


def _mysql_credentials(server_url: str):
    parsed = make_url(server_url)
    if parsed.get_backend_name() != "mysql":
        raise RuntimeError(f"Unsupported server URL for MySQL import: {server_url}")

    return {
        "host": parsed.host or "localhost",
        "port": int(parsed.port or 3306),
        "user": parsed.username or "root",
        "password": parsed.password or "",
    }


def _create_temp_mysql_engine(server_url: str):
    creds = _mysql_credentials(server_url)
    temp_db = f"reporti_upload_{uuid.uuid4().hex[:12]}"

    admin_conn = pymysql.connect(
        host=creds["host"],
        port=creds["port"],
        user=creds["user"],
        password=creds["password"],
        charset="utf8mb4",
        autocommit=True,
    )
    try:
        with admin_conn.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE `{temp_db}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
    finally:
        admin_conn.close()

    temp_url = URL.create(
        "mysql+pymysql",
        username=creds["user"],
        password=creds["password"] or None,
        host=creds["host"],
        port=creds["port"],
        database=temp_db,
        query={"charset": "utf8mb4"},
    )
    engine = create_engine(temp_url)

    return engine, {
        "kind": "uploaded_files_mysql",
        "server": f"{creds['host']}:{creds['port']}",
        "temp_db": temp_db,
    }


def _find_mysql_client() -> str | None:
    candidates = [
        os.getenv("MYSQL_CLIENT_PATH"),
        shutil.which("mysql"),
        r"C:\xampp\mysql\bin\mysql.exe",
        r"C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe",
        r"C:\Program Files\MySQL\MySQL Server 8.4\bin\mysql.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _normalize_sql_dump_for_mysql(sql_text: str, temp_db: str) -> str:
    sql_text = re.sub(r"/\*.*?\*/", "", sql_text, flags=re.S)
    sql_text = re.sub(r"(?m)^\s*--.*$", "", sql_text)
    sql_text = re.sub(r"(?m)^\s*#.*$", "", sql_text)
    sql_text = re.sub(r"(?im)^\s*CREATE\s+DATABASE\s+.*?;\s*$", "", sql_text)
    sql_text = re.sub(r"(?im)^\s*DROP\s+DATABASE\s+.*?;\s*$", "", sql_text)
    sql_text = re.sub(r"(?im)^\s*USE\s+.*?;\s*$", f"USE `{temp_db}`;", sql_text)
    sql_text = re.sub(r"(?im)^\s*LOCK\s+TABLES\s+.*?;\s*$", "", sql_text)
    sql_text = re.sub(r"(?im)^\s*UNLOCK\s+TABLES\s*;\s*$", "", sql_text)
    return sql_text.strip()


def _import_sql_dump_to_mysql(engine, dump_path: Path, temp_db: str, creds: dict):
    mysql_client = _find_mysql_client()
    dump_text = dump_path.read_text(encoding="utf-8", errors="ignore")
    dump_text = _normalize_sql_dump_for_mysql(dump_text, temp_db)

    if mysql_client:
        cmd = [
            mysql_client,
            "--host",
            creds["host"],
            "--port",
            str(creds["port"]),
            "--user",
            creds["user"],
            "--default-character-set=utf8mb4",
            "--force",
            "--database",
            temp_db,
        ]
        if creds["password"]:
            cmd.append(f"--password={creds['password']}")

        proc = subprocess.run(
            cmd,
            input=dump_text.encode("utf-8", errors="ignore"),
            capture_output=True,
        )

        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
            logger.warning("mysql client returned %s for %s: %s", proc.returncode, dump_path.name, stderr)
    else:
        logger.warning(
            "mysql client not found; falling back to direct execution for %s",
            dump_path.name,
        )
        with engine.begin() as conn:
            for statement in [s.strip() for s in dump_text.split(";") if s.strip()]:
                try:
                    conn.exec_driver_sql(statement)
                except Exception as e:
                    logger.debug("Skipped SQL statement during fallback import: %s", e)

    tables = inspect(engine).get_table_names()
    if not tables:
        raise RuntimeError(
            f"Could not import SQL dump {dump_path.name}. No tables were created in MySQL."
        )

    logger.info("Imported SQL dump %s into MySQL (%s tables)", dump_path.name, len(tables))


def _import_sqlite_file_to_mysql(engine, file_path: Path, used_names: set[str]):
    source_engine = create_engine(f"sqlite:///{file_path}")
    inspector = inspect(source_engine)
    for table_name in inspector.get_table_names():
        target_name = _unique_name(_sanitize_table_name(table_name), used_names)
        df = pd.read_sql_table(table_name, source_engine)
        df.to_sql(target_name, engine, if_exists="replace", index=False)
        logger.info("Imported SQLite table %s as %s", table_name, target_name)


def _import_csv_to_mysql(engine, file_path: Path, used_names: set[str]):
    table_name = _unique_name(_sanitize_table_name(file_path.stem), used_names)
    df = pd.read_csv(file_path)
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    logger.info("Imported CSV %s as %s", file_path.name, table_name)


def _import_excel_to_mysql(engine, file_path: Path, used_names: set[str]):
    sheets = pd.read_excel(file_path, sheet_name=None)
    for sheet_name, df in sheets.items():
        table_base = _sanitize_table_name(f"{file_path.stem}_{sheet_name}")
        table_name = _unique_name(table_base, used_names)
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        logger.info("Imported Excel sheet %s from %s as %s", sheet_name, file_path.name, table_name)


def _import_uploaded_files_to_mysql(files: list[Path], source_database_url: str | None = None):
    server_url = _mysql_server_url(source_database_url)
    engine, source_info = _create_temp_mysql_engine(server_url)
    creds = _mysql_credentials(server_url)
    used_names: set[str] = set()

    with engine.begin():
        for file_path in files:
            suffix = file_path.suffix.lower()
            if suffix in SQL_EXTENSIONS:
                _import_sql_dump_to_mysql(engine, file_path, source_info["temp_db"], creds)
            elif suffix in SQLITE_EXTENSIONS:
                _import_sqlite_file_to_mysql(engine, file_path, used_names)
            elif suffix in CSV_EXTENSIONS:
                _import_csv_to_mysql(engine, file_path, used_names)
            elif suffix in EXCEL_EXTENSIONS:
                _import_excel_to_mysql(engine, file_path, used_names)
            else:
                raise RuntimeError(
                    f"Unsupported file type: {file_path.suffix}. "
                    "Supported types are SQL, SQLite, CSV, and Excel files."
                )

    if not inspect(engine).get_table_names():
        raise RuntimeError("Uploaded files did not produce any tables.")

    return engine, source_info


def build_engine_from_source(
    database_url: Optional[str] = None,
    uploaded_files: Optional[Iterable] = None,
    temp_root: Optional[Path] = None,
) -> Tuple[object, dict]:
    """Return a SQLAlchemy engine and metadata for the provided source."""
    if uploaded_files is None:
        raw_files = []
    elif isinstance(uploaded_files, (str, Path)):
        raw_files = [uploaded_files]
    else:
        raw_files = list(uploaded_files)

    files = [_as_path(item) for item in raw_files]

    if files:
        try:
            return _import_uploaded_files_to_mysql(files, source_database_url=database_url)
        except Exception as e:
            has_sql = any(path.suffix.lower() in SQL_EXTENSIONS for path in files)
            if has_sql:
                raise RuntimeError(
                    f"Failed to import uploaded SQL dump(s) into MySQL: {e}"
                ) from e

            logger.warning("MySQL upload import failed, falling back to SQLite: %s", e)
            return _import_uploaded_files_to_sqlite(files, temp_root=temp_root)

    if database_url:
        engine = create_engine(database_url)
        return engine, {
            "kind": "connection_string",
            "label": database_url,
        }

    raise RuntimeError(
        "No database source provided. Upload a file or enter a database connection string."
    )


def _import_uploaded_files_to_sqlite(files: list[Path], temp_root: Optional[Path] = None):
    temp_root = temp_root or Path(tempfile.mkdtemp(prefix="reporti_db_"))
    temp_root.mkdir(parents=True, exist_ok=True)
    db_path = temp_root / "uploaded.sqlite"
    engine = create_engine(f"sqlite:///{db_path}")
    used_names: set[str] = set()

    with engine.begin() as conn:
        for file_path in files:
            suffix = file_path.suffix.lower()
            if suffix in SQLITE_EXTENSIONS:
                source_engine = create_engine(f"sqlite:///{file_path}")
                inspector = inspect(source_engine)
                for table_name in inspector.get_table_names():
                    target_name = _unique_name(_sanitize_table_name(table_name), used_names)
                    df = pd.read_sql_table(table_name, source_engine)
                    df.to_sql(target_name, conn, if_exists="replace", index=False)
            elif suffix in CSV_EXTENSIONS:
                table_name = _unique_name(_sanitize_table_name(file_path.stem), used_names)
                df = pd.read_csv(file_path)
                df.to_sql(table_name, conn, if_exists="replace", index=False)
            elif suffix in EXCEL_EXTENSIONS:
                sheets = pd.read_excel(file_path, sheet_name=None)
                for sheet_name, df in sheets.items():
                    table_base = _sanitize_table_name(f"{file_path.stem}_{sheet_name}")
                    table_name = _unique_name(table_base, used_names)
                    df.to_sql(table_name, conn, if_exists="replace", index=False)
            elif suffix in SQL_EXTENSIONS:
                raise RuntimeError("SQL dumps require a MySQL server. Please make sure MySQL is available.")
            else:
                raise RuntimeError(
                    f"Unsupported file type: {file_path.suffix}. "
                    "Supported types are SQL, SQLite, CSV, and Excel files."
                )

    if not inspect(engine).get_table_names():
        raise RuntimeError("Uploaded files did not produce any tables.")

    return engine, {
        "kind": "uploaded_files_sqlite",
        "sqlite_path": str(db_path),
    }
