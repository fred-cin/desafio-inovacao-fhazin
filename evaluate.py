"""Mede a qualidade da classificação contra rótulos de referência feitos à mão.

Uso:
    python evaluate.py                       # avalia data/processed_reviews.json
    python evaluate.py --engine heuristica   # roda a heurística agora e avalia
    python evaluate.py --engine ia           # roda a IA agora (precisa de chave) e avalia

Salva o resultado em docs/avaliacao.json (lido pela aba "Como funciona" do painel)
e imprime uma tabela em Markdown pronta para colar na documentação.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

import pandas as pd

from dcl import analyzer, llm
from dcl.data import EVAL_PATH, GOLD_PATH, PROCESSED_PATH, RAW_PATH, load_processed, read_reviews_csv
from dcl.taxonomy import CATEGORY_NAMES, SENTIMENTS, normalize_category, normalize_sentiment, to_bool


def parse_aspects(cell: str) -> set[tuple[str, str]]:
    out = set()
    for part in str(cell).split(";"):
        if ":" in part:
            cat, sent = part.rsplit(":", 1)
            out.add((normalize_category(cat), normalize_sentiment(sent)))
    return out


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(pred: pd.DataFrame, gold: pd.DataFrame) -> dict:
    df = gold.merge(pred, on="id", suffixes=("_gold", "_pred"))
    n = len(df)

    # Sentimento geral: acurácia + F1 macro.
    acc = float((df["sentiment_gold"] == df["sentiment_pred"]).mean())
    f1s = []
    for s in SENTIMENTS:
        tp = int(((df.sentiment_gold == s) & (df.sentiment_pred == s)).sum())
        fp = int(((df.sentiment_gold != s) & (df.sentiment_pred == s)).sum())
        fn = int(((df.sentiment_gold == s) & (df.sentiment_pred != s)).sum())
        f1s.append(prf(tp, fp, fn)[2])

    # Categorias (multi-rótulo) e aspecto completo (categoria + sentimento), micro-média.
    tp_c = fp_c = fn_c = tp_a = fp_a = fn_a = 0
    per_cat = {c: [0, 0, 0] for c in CATEGORY_NAMES}
    for _, row in df.iterrows():
        g, p = row["aspects_gold"], row["aspects_pred"]
        gc, pc = {c for c, _ in g}, {c for c, _ in p}
        tp_c += len(gc & pc); fp_c += len(pc - gc); fn_c += len(gc - pc)
        tp_a += len(g & p); fp_a += len(p - g); fn_a += len(g - p)
        for c in CATEGORY_NAMES:
            per_cat[c][0] += c in gc and c in pc
            per_cat[c][1] += c in pc and c not in gc
            per_cat[c][2] += c in gc and c not in pc

    lg, lp = df["legal_gold"], df["legal_pred"]
    legal_acc = float((lg == lp).mean())
    legal_p, legal_r, legal_f = prf(int((lg & lp).sum()), int((~lg & lp).sum()), int((lg & ~lp).sum()))

    return {
        "n": n,
        "sentiment_accuracy": acc,
        "sentiment_macro_f1": sum(f1s) / len(f1s),
        "category_micro": dict(zip(["precision", "recall", "f1"], prf(tp_c, fp_c, fn_c))),
        "aspect_micro": dict(zip(["precision", "recall", "f1"], prf(tp_a, fp_a, fn_a))),
        "legal_risk": {"accuracy": legal_acc, "precision": legal_p, "recall": legal_r, "f1": legal_f},
        "per_category_f1": {c: prf(*v)[2] for c, v in per_cat.items() if sum(v)},
    }


def load_gold() -> pd.DataFrame:
    gold = pd.read_csv(GOLD_PATH, dtype=str)
    gold["aspects"] = gold["aspects"].apply(parse_aspects)
    gold["legal"] = gold["legal_risk"].apply(to_bool)
    return gold[["id", "sentiment", "aspects", "legal"]]


def to_pred_frame(records: list[dict] | pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["aspects"] = df["aspects"].apply(lambda a: {(x["category"], x["sentiment"]) for x in a})
    df["legal"] = df["legal_risk"].astype(bool)
    return df[["id", "sentiment", "aspects", "legal", "engine"]]


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["arquivo", "heuristica", "ia"], default="arquivo")
    args = ap.parse_args()

    if args.engine == "arquivo":
        processed, _ = load_processed(PROCESSED_PATH)
        pred = to_pred_frame(processed)
    else:
        if args.engine == "ia" and not llm.is_available():
            print("Erro: --engine ia exige OPENROUTER_API_KEY.")
            return 1
        raw = read_reviews_csv(RAW_PATH)
        analyses = analyzer.analyze_many(raw["text"], use_llm=args.engine == "ia")
        pred = to_pred_frame([{**a, "id": i} for i, a in zip(raw["id"], analyses)])

    gold = load_gold().rename(columns={"legal": "legal_gold"})
    res = evaluate(pred.rename(columns={"legal": "legal_pred"}), gold)
    engine = ", ".join(sorted(pred["engine"].unique()))
    res.update({"engine": engine, "evaluated_at": datetime.now().isoformat(timespec="seconds")})

    history = []
    if EVAL_PATH.exists():
        history = [h for h in json.loads(EVAL_PATH.read_text(encoding="utf-8")) if h.get("engine") != engine]
    EVAL_PATH.parent.mkdir(exist_ok=True)
    EVAL_PATH.write_text(json.dumps(history + [res], ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nMotor: {engine}  |  {res['n']} avaliações com rótulo de referência\n")
    print("| Métrica | Resultado |\n|---|---|")
    print(f"| Sentimento geral — acurácia | {pct(res['sentiment_accuracy'])} |")
    print(f"| Sentimento geral — F1 macro | {pct(res['sentiment_macro_f1'])} |")
    print(f"| Categorias citadas — F1 | {pct(res['category_micro']['f1'])} |")
    print(f"| Aspecto completo (categoria + sentimento) — F1 | {pct(res['aspect_micro']['f1'])} |")
    print(f"| Risco jurídico — recall | {pct(res['legal_risk']['recall'])} |")
    print(f"| Risco jurídico — precisão | {pct(res['legal_risk']['precision'])} |")
    print(f"\nSalvo em {EVAL_PATH.relative_to(EVAL_PATH.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
