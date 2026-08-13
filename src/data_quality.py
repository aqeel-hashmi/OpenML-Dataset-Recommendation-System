from __future__ import annotations

import json
import re
from datetime import datetime

import pandas as pd

from . import config
from .data_loading import build_document, clean_text, load_raw

MD5_RE = re.compile(r"^[a-f0-9]{32}$", re.IGNORECASE)
URL_RE = re.compile(r"^https?://[^\s]+$")

CRITICAL_FIELDS = [
    "dataset_id", "name", "description", "format", "url",
    "upload_date", "default_target_attribute", "creator", "tags",
]


def _completeness(df: pd.DataFrame) -> dict:
    out = {}
    for f in CRITICAL_FIELDS:
        if f == "tags":
            non_missing = (df["n_tags"] > 0).mean()
        elif f == "description":
            non_missing = (df["desc_len"] > 0).mean()
        else:
            col = df[f]
            non_missing = col.notna().mean()
            if col.dtype == object:
                non_missing = col.apply(
                    lambda v: v not in (None, "", [], "None")
                ).mean()
        out[f] = round(float(non_missing), 4)
    return out


def _uniqueness(df: pd.DataFrame) -> dict:
    n = len(df)
    dup_ids = int(df["dataset_id"].duplicated().sum())
    dup_md5 = int(df["md5_checksum"].duplicated(keep=False).sum())
    dup_name_ver = int(df.duplicated(subset=["name", "version"]).sum())
    return {
        "n_records": n,
        "duplicate_dataset_ids": dup_ids,
        "records_sharing_md5": dup_md5,
        "duplicate_name_version_pairs": dup_name_ver,
        "unique_names": int(df["name"].nunique()),
        "dataset_id_uniqueness_rate": round(1 - dup_ids / n, 4),
    }


def _validity(df: pd.DataFrame) -> dict:
    md5_valid = df["md5_checksum"].astype(str).apply(lambda s: bool(MD5_RE.match(s))).mean()
    url_valid = df["url"].astype(str).apply(lambda s: bool(URL_RE.match(s))).mean()
    version_valid = ((df["version"].fillna(-1) > 0)).mean()

    now = pd.Timestamp(datetime(2026, 12, 31))
    date_valid = df["upload_date"].apply(
        lambda d: pd.notna(d) and pd.Timestamp("2010-01-01") <= d <= now
    ).mean()
    return {
        "md5_valid_rate": round(float(md5_valid), 4),
        "url_valid_rate": round(float(url_valid), 4),
        "version_positive_rate": round(float(version_valid), 4),
        "upload_date_plausible_rate": round(float(date_valid), 4),
    }


def _consistency(df: pd.DataFrame) -> dict:
    raw_formats = df["format"].astype(str).value_counts().to_dict()
    norm_formats = df["format_norm"].value_counts().to_dict()
    return {
        "raw_format_variants": len(raw_formats),
        "raw_format_distribution": {k: int(v) for k, v in raw_formats.items()},
        "normalised_format_variants": len(norm_formats),
        "normalised_format_distribution": {k: int(v) for k, v in norm_formats.items()},
    }


def _conformity(df: pd.DataFrame) -> dict:
    tags_listy = df["tags"].apply(lambda x: isinstance(x, list)).mean()
    avg_tags = float(df.loc[df["n_tags"] > 0, "n_tags"].mean())
    return {
        "tags_parse_rate": round(float(tags_listy), 4),
        "avg_tags_when_present": round(avg_tags, 2),
        "tagged_fraction": round(float((df["n_tags"] > 0).mean()), 4),
    }


def _timeliness(df: pd.DataFrame) -> dict:
    valid = df["upload_date"].dropna()
    return {
        "earliest_upload": str(valid.min().date()) if len(valid) else None,
        "latest_upload": str(valid.max().date()) if len(valid) else None,
        "median_upload_year": int(valid.dt.year.median()) if len(valid) else None,
    }


def compute_quality_report(df: pd.DataFrame) -> dict:
    report = {
        "completeness": _completeness(df),
        "uniqueness": _uniqueness(df),
        "validity": _validity(df),
        "consistency": _consistency(df),
        "conformity": _conformity(df),
        "timeliness": _timeliness(df),
    }

    completeness_score = sum(report["completeness"].values()) / len(report["completeness"])
    validity_score = sum(report["validity"].values()) / len(report["validity"])
    uniqueness_score = report["uniqueness"]["dataset_id_uniqueness_rate"]
    conformity_score = report["conformity"]["tags_parse_rate"]
    report["composite_score"] = round(
        (completeness_score + validity_score + uniqueness_score + conformity_score) / 4, 4
    )
    report["dimension_scores"] = {
        "completeness": round(completeness_score, 4),
        "validity": round(validity_score, 4),
        "uniqueness": round(uniqueness_score, 4),
        "conformity": round(conformity_score, 4),
    }
    return report


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:

    log = {}
    n0 = len(df)


    df = df[df["dataset_id"].notna()].copy()
    log["dropped_missing_id"] = n0 - len(df)


    df = df.sort_values(["dataset_id", "version"]).drop_duplicates(
        subset="dataset_id", keep="last"
    )
    log["dropped_duplicate_ids"] = n0 - log["dropped_missing_id"] - len(df)


    df["clean_name"] = df["name"].apply(clean_text)
    df["clean_description"] = df["description"].apply(clean_text)
    df["document"] = df.apply(build_document, axis=1)
    df["doc_len"] = df["document"].str.len()

    before = len(df)
    df = df[df["doc_len"] >= 15].copy()
    log["dropped_low_text"] = before - len(df)

    df = df.reset_index(drop=True)
    df["row_idx"] = df.index

    log["final_rows"] = len(df)
    log["initial_rows"] = n0
    return df, log


def run(save: bool = True) -> tuple[pd.DataFrame, dict]:
    raw = load_raw()
    report = compute_quality_report(raw)
    clean, remediation = clean_dataset(raw)
    report["remediation"] = remediation
    # Re-assess key metrics after cleaning to show improvement.
    report["post_clean_completeness"] = _completeness(clean)

    if save:
        out_cols = [
            "row_idx", "dataset_id", "name", "clean_name", "description",
            "clean_description", "document", "doc_len", "tags", "n_tags",
            "creator_str", "default_target_attribute", "format_norm",
            "version", "upload_date", "url",
        ]
        clean[out_cols].to_parquet(config.CLEAN_DATA_PATH, index=False)
        with open(config.METRIC_DIR / "data_quality_report.json", "w") as fh:
            json.dump(report, fh, indent=2, default=str)

    return clean, report


if __name__ == "__main__":
    clean, report = run()
    print(json.dumps({k: v for k, v in report.items()
                      if k in ("dimension_scores", "composite_score", "remediation")},
                     indent=2, default=str))
    print(f"\nClean dataset: {len(clean)} rows -> {config.CLEAN_DATA_PATH}")
