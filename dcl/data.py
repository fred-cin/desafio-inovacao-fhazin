"""Entrada e saída de dados: CSV bruto, base processada e formatos derivados."""

from __future__ import annotations

import io
import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .taxonomy import fold

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw_reviews.csv"
PROCESSED_PATH = DATA_DIR / "processed_reviews.json"
GOLD_PATH = DATA_DIR / "rotulos_referencia.csv"
TEMPLATE_PATH = DATA_DIR / "modelo_importacao.csv"
EVAL_PATH = ROOT / "docs" / "avaliacao.json"

# Nomes de coluna aceitos na importação (planilhas reais vêm com todo tipo de cabeçalho).
COLUMN_ALIASES = {
    "text": ["text", "texto", "avaliacao", "comentario", "review", "reclamacao", "descricao"],
    "company": ["company", "empresa", "construtora", "incorporadora", "empreendimento"],
    "source": ["source", "fonte", "origem", "canal", "plataforma"],
    "date": ["date", "data", "data_avaliacao", "criado_em", "created_at"],
    "id": ["id", "codigo", "review_id"],
}


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    folded = {fold(c).replace(" ", "_"): c for c in df.columns}
    for canon, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in folded:
                mapping[folded[alias]] = canon
                break
    return df.rename(columns=mapping)


def parse_dates(values: pd.Series) -> pd.Series:
    """ISO (2026-11-15) primeiro; depois o formato brasileiro (15/11/2026). Inválidas viram hoje."""
    parsed = pd.to_datetime(values, format="%Y-%m-%d", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed[missing] = pd.to_datetime(values[missing], dayfirst=True, errors="coerce")
    return parsed.dt.date.fillna(date.today())


def read_reviews_csv(source, default_company: str = "Minha construtora", default_source: str = "Importado") -> pd.DataFrame:
    """Lê um CSV de avaliações, tolerante a cabeçalhos e separadores diferentes."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    df = pd.read_csv(source, sep=None, engine="python", encoding="utf-8-sig", dtype=str)
    df = _rename_columns(df)
    if "text" not in df.columns:
        raise ValueError("o arquivo precisa de uma coluna de texto (ex.: 'texto', 'avaliacao' ou 'comentario')")
    df["text"] = df["text"].fillna("").str.strip()
    df = df[df["text"].str.len() > 0].copy()
    df["company"] = df["company"].fillna(default_company) if "company" in df else default_company
    df["source"] = df["source"].fillna(default_source) if "source" in df else default_source
    if "date" in df:
        df["date"] = parse_dates(df["date"])
    else:
        df["date"] = date.today()
    if "id" not in df:
        df["id"] = [f"imp-{i:04d}" for i in range(1, len(df) + 1)]
    return df[["id", "company", "source", "date", "text"]].reset_index(drop=True)


def records_from_analysis(raw: pd.DataFrame, analyses: list[dict]) -> list[dict]:
    out = []
    for row, analysis in zip(raw.to_dict("records"), analyses):
        d = row["date"]
        out.append({**row, "date": d.isoformat() if hasattr(d, "isoformat") else str(d), **analysis})
    return out


def save_processed(records: list[dict], path: Path = PROCESSED_PATH, demo: bool = True) -> None:
    engines = sorted({r.get("engine", "?") for r in records})
    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "engines": engines,
            "demo_data": demo,
            "count": len(records),
        },
        "reviews": records,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_processed(path: Path = PROCESSED_PATH) -> tuple[pd.DataFrame, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    reviews = payload["reviews"] if isinstance(payload, dict) else payload
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    return to_frame(reviews), meta


def to_frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["categories"] = df["aspects"].apply(lambda a: [x["category"] for x in a])
    return df


def explode_aspects(df: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por (avaliação, aspecto). Base de todos os gráficos por categoria."""
    long = df[["id", "company", "source", "date", "aspects"]].explode("aspects").dropna(subset=["aspects"])
    long["category"] = long["aspects"].apply(lambda a: a["category"])
    long["aspect_sentiment"] = long["aspects"].apply(lambda a: a["sentiment"])
    return long.drop(columns="aspects").reset_index(drop=True)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    out = df.copy()
    if "aspects" in out:
        out["aspects"] = out["aspects"].apply(lambda a: "; ".join(f"{x['category']}: {x['sentiment']}" for x in a))
    out = out.drop(columns=[c for c in ("categories",) if c in out])
    return out.to_csv(index=False).encode("utf-8-sig")
