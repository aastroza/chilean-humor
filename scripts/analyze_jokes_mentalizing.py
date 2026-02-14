#!/usr/bin/env python3
"""CLI wrapper for joke mentalizing analysis pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chilean_humor_processing.analyze_jokes_mentalizing import main


if __name__ == "__main__":
    main()
