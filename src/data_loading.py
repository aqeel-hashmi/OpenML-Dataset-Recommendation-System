from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from . import config


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []

        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s.replace("'", '"'))
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed if str(v).strip()]
            except json.JSONDecodeError:
                pass
        return [s]
    return [str(value).strip()]


def load_raw(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as fh:
        records = json.load(fh)

    df = pd.DataFrame(records)


    df["tags"] = df["tag"].apply(_as_list)
    df["creator_list"] = df["creator"].apply(_as_list)
    df["creator_str"] = df["creator_list"].apply(lambda xs: ", ".join(xs))

    df["dataset_id"] = pd.to_numeric(df["dataset_id"], errors="coerce").astype("Int64")
    df["version"] = pd.to_numeric(df["version"], errors="coerce").astype("Int64")
    df["upload_date"] = pd.to_datetime(df["upload_date"], errors="coerce")

    df["format_norm"] = (
        df["format"].astype(str).str.lower().str.replace("sparse_arff", "sparse_arff")
    )
    df["description"] = df["description"].fillna("").astype(str)
    df["name"] = df["name"].fillna("").astype(str)
    df["default_target_attribute"] = (
        df["default_target_attribute"].fillna("").astype(str)
    )

    df["n_tags"] = df["tags"].apply(len)
    df["desc_len"] = df["description"].str.len()
    df["has_description"] = df["desc_len"] > 0

    return df


_WS_RE = re.compile(r"\s+")
_MD_RE = re.compile(r"[*_`#>\[\]()]+")
_URL_RE = re.compile(r"https?://\S+")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = _URL_RE.sub(" ", text)
    text = _MD_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def build_document(row: pd.Series, fields=None) -> str:
    fields = fields or config.TEXT_FIELDS
    parts = []
    for f in fields:
        val = row.get(f, "")
        if isinstance(val, list):
            val = " ".join(val)
        val = clean_text(str(val))
        if val:
            parts.append(val)
    return " . ".join(parts)

if __name__ == "__main__":
    df = load_raw()
    print(f"Loaded {len(df)} records, {df.shape[1]} columns")
    print(df[["dataset_id", "name", "n_tags", "desc_len", "format_norm"]].head())
