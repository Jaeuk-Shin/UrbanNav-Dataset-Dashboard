"""Allow ``streamlit run __main__.py`` from inside the dashboard/ directory."""

import sys
from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import main

main()
