"""
Single-entry launcher for the AI Database Assistant frontend.
"""
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from gradio_ui import launch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 50)
    logger.info("Starting AI Database Assistant...")
    logger.info("=" * 50)

    logger.info("Interface ready at: http://127.0.0.1:7860")
    print("OPEN THIS IN YOUR BROWSER: http://127.0.0.1:7860")
    launch(host="127.0.0.1", port=7860, share=False, quiet=False)


if __name__ == "__main__":
    main()
