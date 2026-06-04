"""
AI Database Assistant - Natural Language to SQL
"""
from .ai_init import initialize
from .pipeline import DatabaseAssistant, run_pipeline
from .gradio_ui import launch

__version__ = "1.0.0"
__all__ = ["initialize", "DatabaseAssistant", "run_pipeline", "launch"]