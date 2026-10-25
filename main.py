"""
Vector Database with Qdrant — Hyperskill project entry map.

Each stage is a separate script so earlier work is never overwritten:

  poetry run python stage_1.py   # load arXiv embeddings into Qdrant
  poetry run python stage_2.py   # print top-5 papers similar to 1311.5068
  poetry run python stage_3.py   # (later) natural-language search via OpenAI
  poetry run python stage_4.py   # (later) filters / keys
  poetry run python stage_5.py   # (later) FastAPI wrap

Shared client/config: common.py

For Hyperskill "Check", run the stage_N.py file for the stage you are on
(with Qdrant still running on localhost:6333).
"""

from __future__ import annotations


def main() -> None:
    print(__doc__)


if __name__ == "__main__":
    main()
