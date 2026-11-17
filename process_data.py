"""Pipeline de processamento: avaliações brutas (CSV) -> base estruturada (JSON).

Uso:
    export OPENROUTER_API_KEY=sk-or-...        # opcional; sem chave usa a heurística local
    python process_data.py                     # processa data/raw_reviews.csv
    python process_data.py --input minhas.csv --engine ia --demo-data no
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from dcl import analyzer, llm
from dcl.data import PROCESSED_PATH, RAW_PATH, read_reviews_csv, records_from_analysis, save_processed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(RAW_PATH))
    p.add_argument("--output", default=str(PROCESSED_PATH))
    p.add_argument("--engine", choices=["auto", "ia", "heuristica"], default="auto")
    p.add_argument("--demo-data", choices=["sim", "no"], default="sim", help="marca a base como dados simulados")
    args = p.parse_args()

    if args.engine == "ia" and not llm.is_available():
        print("Erro: --engine ia exige OPENROUTER_API_KEY (variável de ambiente ou .streamlit/secrets.toml).")
        return 1
    use_llm = {"auto": None, "ia": True, "heuristica": False}[args.engine]

    raw = read_reviews_csv(args.input)
    motor = llm.get_model() if (use_llm or (use_llm is None and llm.is_available())) else "heurística local"
    print(f"Processando {len(raw)} avaliações com {motor}...")

    def progress(i: int, n: int) -> None:
        print(f"\r  {i}/{n}", end="", flush=True)

    analyses = analyzer.analyze_many(raw["text"], use_llm=use_llm, on_progress=progress)
    print()
    for a in analyses:
        if a.get("warning"):
            print("  aviso:", a["warning"])

    records = records_from_analysis(raw, analyses)
    from pathlib import Path

    save_processed(records, Path(args.output), demo=args.demo_data == "sim")
    print(f"Salvo em {args.output}")
    print("Sentimentos:", dict(Counter(r["sentiment"] for r in records)))
    print("Motores:", dict(Counter(r["engine"] for r in records)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
