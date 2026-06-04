"""
Gradio frontend for AI Database Assistant.
"""
import logging
import os

import gradio as gr
import pandas as pd

from ai_init import initialize
from pipeline import DatabaseAssistant
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine.url import URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_response(result):
    """Format pipeline result for Gradio display."""
    if not result:
        return "No response", "", None

    answer = result.get("answer", "No answer generated")
    if result.get("error"):
        answer += f"\n\nWarning: {result['error']}"

    sql = result.get("sql")
    sql_display = f"```sql\n{sql}\n```" if sql else "No SQL generated"

    rows = result.get("rows", [])
    table = pd.DataFrame(rows) if rows else None

    timings = result.get("timings") or {}
    if timings:
        timing_bits = []
        for key in [
            "retrieve_initial",
            "normalize",
            "rag_expand",
            "schema_build",
            "sql_generate",
            "sql_validate",
            "sql_execute",
            "answer",
            "total_seconds",
        ]:
            if key in timings:
                timing_bits.append(f"{key}: {timings[key]}s")
        if timing_bits:
            answer += "\n\nTiming: " + " | ".join(timing_bits)

    return answer, sql_display, table


def _source_status(init_data):
    source_info = init_data.get("source_info", {}) if init_data else {}
    kind = source_info.get("kind", "unknown")
    if kind == "connection_string":
        label = source_info.get("label", "database connection")
        return f"Loaded database connection: `{label}`"
    if kind == "direct_mysql_database":
        server = source_info.get("server", "MySQL server")
        database = source_info.get("database", "")
        return (
            "Connected directly to an existing MySQL database.\n\n"
            f"Server: `{server}`\n\n"
            f"Database: `{database}`"
        )
    if kind == "uploaded_files_mysql":
        server = source_info.get("server", "MySQL server")
        temp_db = source_info.get("temp_db", "")
        return (
            "Loaded uploaded files into a temporary MySQL database.\n\n"
            f"Server: `{server}`\n\n"
            f"Temporary database: `{temp_db}`"
        )
    if kind == "uploaded_files":
        files = source_info.get("files", [])
        sqlite_path = source_info.get("sqlite_path", "")
        return (
            f"Loaded {len(files)} uploaded file(s) into a temporary SQLite database.\n\n"
            f"`{sqlite_path}`"
        )
    return "No database loaded yet. Upload files or enter a connection string."


def _build_mysql_server_url(host, port, user, password):
    return str(
        URL.create(
            "mysql+pymysql",
            username=(user or "root").strip() or "root",
            password=(password or "").strip() or None,
            host=(host or "localhost").strip() or "localhost",
            port=int(port or 3306),
        )
    )


def _build_mysql_database_url(host, port, user, password, database):
    return str(
        URL.create(
            "mysql+pymysql",
            username=(user or "root").strip() or "root",
            password=(password or "").strip() or None,
            host=(host or "localhost").strip() or "localhost",
            port=int(port or 3306),
            database=(database or "").strip(),
        )
    )


def create_interface(default_database_url=None):
    """Create and return the Gradio interface."""
    default_database_url = default_database_url or os.getenv("DATABASE_URL", "")

    def refresh_mysql_databases(mysql_host, mysql_port, mysql_user, mysql_password):
        server_url = _build_mysql_server_url(mysql_host, mysql_port, mysql_user, mysql_password)
        try:
            engine = create_engine(server_url)
            inspector = inspect(engine)
            databases = [
                name for name in inspector.get_schema_names()
                if name not in {"information_schema", "mysql", "performance_schema", "sys"}
            ]
            databases = sorted(set(databases))
            if not databases:
                return gr.update(choices=[], value=None), "No user databases found on this MySQL server."
            return (
                gr.update(choices=databases, value=databases[0]),
                f"Found {len(databases)} database(s) on the MySQL server.",
            )
        except Exception as e:
            logger.exception("Failed to list MySQL databases")
            return gr.update(choices=[], value=None), f"Could not list databases: {e}"

    def load_database(database_url, uploaded_files, mysql_host, mysql_port, mysql_user, mysql_password, mysql_database):
        database_url = (database_url or "").strip() or None
        uploaded_files = uploaded_files or []
        mysql_database = (mysql_database or "").strip() or None
        direct_mysql_requested = bool(mysql_database and not uploaded_files and not database_url)

        if not database_url and not uploaded_files and not mysql_database:
            return None, "Upload at least one file or provide a connection string.", [], "", None

        if direct_mysql_requested:
            database_url = _build_mysql_database_url(
                mysql_host, mysql_port, mysql_user, mysql_password, mysql_database
            )

        try:
            init_data = initialize(
                database_url=database_url,
                uploaded_files=uploaded_files,
            )
            if direct_mysql_requested:
                init_data["source_info"] = {
                    "kind": "direct_mysql_database",
                    "server": f"{(mysql_host or 'localhost').strip() or 'localhost'}:{int(mysql_port or 3306)}",
                    "database": mysql_database,
                }
            assistant = DatabaseAssistant(init_data)
            inspector = inspect(init_data["engine"])
            table_count = len(inspector.get_table_names())
            view_count = len(inspector.get_view_names())
            total_count = table_count + view_count
            status = (
                _source_status(init_data)
                + f"\n\nTables detected: `{table_count}`"
                + f"\nViews detected: `{view_count}`"
                + f"\nTotal schema objects: `{total_count}`"
            )
            return assistant, status, [], "", None
        except Exception as e:
            logger.exception("Failed to load database source")
            return None, f"Failed to load database: {e}", [], "", None

    def chat_interface(question, history, assistant):
        history = history or []

        if assistant is None:
            warning = "Load a database first using the panel on the left."
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": warning})
            return history, "", None

        result = assistant.run(question)
        answer, sql, table = format_response(result)

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

        return history, sql, table

    with gr.Blocks(title="AI Database Assistant") as demo:
        assistant_state = gr.State(None)

        gr.Markdown("# AI Database Assistant")
        gr.Markdown(
            "Connect to an existing MySQL database, or upload a dump/file. "
            "For enterprise databases, direct MySQL connection is the most reliable path."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Connect to MySQL server")
                mysql_host = gr.Textbox(label="Host", value="localhost")
                mysql_port = gr.Number(label="Port", value=3306, precision=0)
                mysql_user = gr.Textbox(label="User", value="root")
                mysql_password = gr.Textbox(label="Password", type="password", value="")
                mysql_database = gr.Dropdown(label="Database", choices=[], value=None)
                refresh_btn = gr.Button("Refresh Databases")
                connect_btn = gr.Button("Connect to Selected DB", variant="primary")

                gr.Markdown("### Or paste a connection string")
                database_url = gr.Textbox(
                    label="Database URL",
                    value=default_database_url,
                    placeholder="sqlite:///path/to/db.sqlite or mysql+pymysql://user:pass@host/db",
                    lines=2,
                )
                gr.Markdown("### Or upload files")
                uploaded_files = gr.File(
                    label="Upload database files",
                    file_count="multiple",
                    file_types=[".db", ".sqlite", ".sqlite3", ".csv", ".xlsx", ".xls", ".sql"],
                    type="filepath",
                )
                load_btn = gr.Button("Load Database", variant="primary")
                source_status = gr.Markdown(_source_status(None))

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=420)
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask about the uploaded data or ask a general question",
                    lines=2,
                )
                submit_btn = gr.Button("Ask", variant="primary")
                clear_btn = gr.Button("Clear")

                sql_output = gr.Markdown(label="Generated SQL")
                table_output = gr.Dataframe(label="Results", interactive=False)

        refresh_btn.click(
            fn=refresh_mysql_databases,
            inputs=[mysql_host, mysql_port, mysql_user, mysql_password],
            outputs=[mysql_database, source_status],
        )

        connect_btn.click(
            fn=load_database,
            inputs=[database_url, uploaded_files, mysql_host, mysql_port, mysql_user, mysql_password, mysql_database],
            outputs=[assistant_state, source_status, chatbot, sql_output, table_output],
        )

        load_btn.click(
            fn=load_database,
            inputs=[database_url, uploaded_files, mysql_host, mysql_port, mysql_user, mysql_password, mysql_database],
            outputs=[assistant_state, source_status, chatbot, sql_output, table_output],
        )

        submit_btn.click(
            fn=chat_interface,
            inputs=[question_input, chatbot, assistant_state],
            outputs=[chatbot, sql_output, table_output],
        )

        question_input.submit(
            fn=chat_interface,
            inputs=[question_input, chatbot, assistant_state],
            outputs=[chatbot, sql_output, table_output],
        )

        clear_btn.click(
            fn=lambda: ([], "", None),
            inputs=[],
            outputs=[chatbot, sql_output, table_output],
        )

        gr.Examples(
            examples=[
                "Show a summary of the available records",
                "What is the average of the main numeric field?",
                "List the most recent records",
                "Which group has the highest count?",
                "What is SQL injection and why is it dangerous?",
            ],
            inputs=question_input,
            label="Try these examples",
        )

    return demo


def launch(host="127.0.0.1", port=7860, **kwargs):
    """Launch the Gradio interface."""
    demo = create_interface()
    demo.launch(server_name=host, server_port=port, theme=gr.themes.Soft(), **kwargs)


if __name__ == "__main__":
    launch()
