from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, embeddings
from src.recommender import ContentRecommender

st.set_page_config(page_title="OpenML Dataset Recommender", layout="wide",
                   page_icon="🔎")


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not config.CLEAN_DATA_PATH.exists():
        from src import data_quality
        data_quality.run(save=True)
    return pd.read_parquet(config.CLEAN_DATA_PATH)


@st.cache_data(show_spinner=False)
def load_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_resource(show_spinner=True)
def build_engine(rep: str):
    df = load_data()
    docs = df["document"].tolist()
    if rep == "Semantic (transformer)":
        emb, fitted = embeddings.build_semantic(docs)
        embed_fn = embeddings.transform_semantic
    else:
        tuned = load_json(config.METRIC_DIR / "tuning_best.json")
        params = (tuned or {}).get("best_params") if tuned else None
        emb, fitted = embeddings.build_tfidf(docs, params)
        embed_fn = embeddings.transform_tfidf
    rec = ContentRecommender(emb.vectors, ids=df["dataset_id"].values,
                             names=df["name"].values)
    return rec, fitted, embed_fn


def rec_table(df, idx, sims):
    rows = []
    for j, s in zip(idx, sims):
        r = df.iloc[int(j)]
        tags = ", ".join(list(r["tags"])[:6])
        desc = (r["clean_description"] or "")[:200]
        rows.append({"similarity": round(float(s), 3), "dataset_id": int(r["dataset_id"]),
                     "name": r["name"], "tags": tags, "description": desc})
    return pd.DataFrame(rows)


st.sidebar.title("🔎 OpenML Recommender")
st.sidebar.caption("Content-based dataset recommender over 6.3k OpenML datasets.")
rep = st.sidebar.radio("Representation",
                       ["TF-IDF (tuned)", "Semantic (transformer)"], index=0)
k = st.sidebar.slider("Number of recommendations (k)", 3, 25, 10)

df = load_data()
st.sidebar.metric("Datasets in catalogue", f"{len(df):,}")

tabs = st.tabs(["🧭 Recommend", "🔤 Search", "📊 Evaluation",
                "✅ Data Quality", "🛡️ Robustness"])

with tabs[0]:
    st.header("More like this")
    st.write("Pick a dataset and get the most similar datasets by content.")
    options = (df["name"] + "  (id=" + df["dataset_id"].astype(str) + ")").tolist()
    default = int(df.index[df["name"].str.lower() == "iris"][0]) if \
        (df["name"].str.lower() == "iris").any() else 0
    choice = st.selectbox("Query dataset", options, index=default)
    if st.button("Recommend", type="primary"):
        row_idx = options.index(choice)
        rec, fitted, _ = build_engine(rep)
        sel = df.iloc[row_idx]
        st.markdown(f"**{sel['name']}** — _{', '.join(list(sel['tags'])[:8])}_")
        with st.expander("Query description"):
            st.write(sel["clean_description"][:1200] or "(no description)")
        idx, sims = rec.recommend_by_index(row_idx, k=k)
        st.dataframe(rec_table(df, idx, sims), use_container_width=True,
                     hide_index=True)

with tabs[1]:
    st.header("Natural-language dataset search")
    st.write("Describe the data you need; we rank datasets by content similarity.")
    query = st.text_area("Describe the dataset you are looking for",
                         "credit card fraud detection transactions imbalanced classification",
                         height=80)
    if st.button("Search", type="primary"):
        rec, fitted, embed_fn = build_engine(rep)
        qvec = embed_fn(fitted, [query])[0]
        idx, sims = rec.recommend_by_vector(qvec, k=k)
        st.dataframe(rec_table(df, idx, sims), use_container_width=True,
                     hide_index=True)

with tabs[2]:
    st.header("Model comparison & ranking metrics")
    results = load_json(config.METRIC_DIR / "pipeline_results.json")
    if not results:
        st.info("Run `python run_pipeline.py` first to generate evaluation metrics.")
    else:
        st.success(f"Best model: **{results['best_model']}**  ·  "
                   f"runtime {results.get('runtime_seconds','?')}s")
        comp = results["comparison"]
        models = [m for m in comp if not m.startswith("_")]
        for signal in ("topical", "study"):
            st.subheader(f"{signal.capitalize()} relevance")
            table = {m: comp[m][signal] for m in models}
            st.dataframe(pd.DataFrame(table).T, use_container_width=True)
            fig = config.FIGURE_DIR / f"pr_curve_{signal}.png"
            if fig.exists():
                st.image(str(fig))

with tabs[3]:
    st.header("Data quality dashboard")
    dq = load_json(config.METRIC_DIR / "data_quality_report.json")
    if not dq:
        st.info("Run the pipeline to generate the data quality report.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Composite DQ score", dq["composite_score"])
        c2.metric("Clean rows", dq["remediation"]["final_rows"])
        c3.metric("Rows dropped",
                  dq["remediation"]["initial_rows"] - dq["remediation"]["final_rows"])
        st.subheader("Dimension scores")
        st.dataframe(pd.DataFrame([dq["dimension_scores"]]).T.rename(columns={0: "score"}),
                     use_container_width=True)
        fig = config.FIGURE_DIR / "data_quality_completeness.png"
        if fig.exists():
            st.image(str(fig))
        with st.expander("Full report JSON"):
            st.json(dq)

with tabs[4]:
    st.header("Perturbation / robustness analysis")
    results = load_json(config.METRIC_DIR / "pipeline_results.json")
    pert = results.get("perturbation") if results else None
    if not pert:
        st.info("Run the pipeline (with perturbation enabled) to see robustness results.")
    else:
        st.metric("Clean precision@10", pert.get("clean_precision@10"))
        fig = config.FIGURE_DIR / "perturbation_robustness.png"
        if fig.exists():
            st.image(str(fig))
        rows = []
        for kind, levels in pert["perturbations"].items():
            for lvl, m in levels.items():
                rows.append({"perturbation": kind, "level": float(lvl), **m})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
