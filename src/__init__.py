"""Sneaker Sniper: zero-tag visual search & metric learning engine.

Importing the package sets a few process-wide environment defaults that must be in
place *before* torch / faiss / mlflow are imported:

* ``KMP_DUPLICATE_LIB_OK=TRUE`` - faiss-cpu and torch both ship an Intel OpenMP
  runtime on Windows; loading both in one process otherwise aborts with
  "libiomp5md.dll already initialized".
* ``HF_HUB_DISABLE_SYMLINKS_WARNING=1`` - silence the Windows symlink warning of the
  Hugging Face cache (weights are still cached, just copied instead of symlinked).
* ``MLFLOW_DISABLE_AGENT_HINT=1`` - silence an informational banner printed on import.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

__version__ = "1.0.0"
