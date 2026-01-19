#prompt to run
#python3 phaseA_validate_clusters.py \
#  --in ~/Downloads/cfs_pem_symptom_clustered_sample.parquet \
#  --outdir ~/Downloads/cfs_cluster_validation





#!/usr/bin/env python3
import argparse
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity


# --- Artifact heuristics (transparent + editable) ---
ARTIFACT_PATTERNS = {
    "healthcare_navigation": re.compile(r"\b(doctor|doctors|diagnos|test results?|bloodwork|specialist|appointment|insurance|nhs)\b", re.I),
    "life_context": re.compile(r"\b(family|parents|relationship|boyfriend|girlfriend|husband|wife|kids|friends|lonely|suicide)\b", re.I),
    "work_school": re.compile(r"\b(work|job|boss|employer|hr\b|college|school|class|university|semester)\b", re.I),
    "supplements_meds": re.compile(r"\b(ldn|naltrexone|abilify|ssri|snri|propranolol|ivabradine|supplement|vitamin|dose|mg)\b", re.I),
    "meta_reddit": re.compile(r"\b(upvote|downvote|subreddit|mods|reddit)\b", re.I),
}

def artifact_flags(text: str) -> dict:
    t = text or ""
    return {k: bool(rx.search(t)) for k, rx in ARTIFACT_PATTERNS.items()}

def top_terms_for_cluster(X, y, feature_names, k, top_n=20):
    # centroid from mean tf-idf across docs in cluster
    idx = np.where(y == k)[0]
    if len(idx) == 0:
        return []
    centroid = X[idx].mean(axis=0)
    # centroid is 1xV sparse matrix
    arr = np.asarray(centroid).ravel()
    top_idx = np.argsort(arr)[::-1][:top_n]
    return [feature_names[i] for i in top_idx if arr[i] > 0]

def mean_intra_cluster_cosine(X, y, k, sample_cap=200):
    idx = np.where(y == k)[0]
    if len(idx) < 3:
        return np.nan
    if len(idx) > sample_cap:
        idx = np.random.choice(idx, size=sample_cap, replace=False)
    # cosine sim on dense can be heavy; limit sample_cap
    M = X[idx]
    S = cosine_similarity(M)
    # take upper triangle (excluding diagonal)
    triu = S[np.triu_indices_from(S, k=1)]
    return float(np.mean(triu)) if triu.size else np.nan

def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input parquet containing at least: text, cluster OR (text only for re-cluster)")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--cluster-col", default="cluster", help="If present, will analyze existing clusters; else will re-cluster.")
    ap.add_argument("--recluster-k", type=int, default=None, help="If set, ignore existing cluster labels and re-cluster with this K.")
    ap.add_argument("--min-len", type=int, default=120)
    ap.add_argument("--max-features", type=int, default=35000)
    ap.add_argument("--ngram-max", type=int, default=2)
    ap.add_argument("--stability-runs", type=int, default=8)
    ap.add_argument("--outdir", default="cfs_cluster_validation_out")
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    if args.text_col not in df.columns:
        raise SystemExit(f"Missing text column: {args.text_col}. Columns={list(df.columns)}")

    df = df.copy()
    df["text"] = df[args.text_col].fillna("").astype(str)
    df["text_len"] = df["text"].str.len()
    df = df[df["text_len"] >= args.min_len].reset_index(drop=True)

    # Vectorize
    vec = TfidfVectorizer(
        stop_words="english",
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        min_df=2
    )
    X = vec.fit_transform(df["text"])
    feat = np.array(vec.get_feature_names_out())

    # Determine cluster labels to analyze
    if args.recluster_k is not None or args.cluster_col not in df.columns:
        K = int(args.recluster_k) if args.recluster_k else 23
        km = MiniBatchKMeans(n_clusters=K, random_state=0, batch_size=2048, n_init="auto")
        y = km.fit_predict(X)
        df["cluster"] = [f"C{c:02d}" for c in y]
    else:
        df["cluster"] = df[args.cluster_col].astype(str)
        # normalize to C## if it looks like integers
        if df["cluster"].str.match(r"^\d+$").all():
            df["cluster"] = df["cluster"].astype(int).map(lambda c: f"C{c:02d}")
        K = df["cluster"].nunique()

    # Artifact flags
    flags = df["text"].apply(artifact_flags).apply(pd.Series)
    for col in flags.columns:
        df[f"art_{col}"] = flags[col].astype(bool)

    # Coherence + top terms
    # Convert to numeric cluster ids for convenience
    cluster_names = sorted(df["cluster"].unique())
    name_to_id = {n: i for i, n in enumerate(cluster_names)}
    y_id = df["cluster"].map(name_to_id).to_numpy()

    # Stability: rerun clustering multiple times on same X, compute ARI vs run0
    # NOTE: This is a *sanity check* for whether “clusters exist at all” in this representation,
    # not a proof of biological reality.
    stability = {"ari_mean_vs_run0": np.nan, "ari_runs": []}
    if args.stability_runs >= 2:
        Ks = len(cluster_names)
        ys = []
        for seed in range(args.stability_runs):
            km = MiniBatchKMeans(n_clusters=Ks, random_state=seed, batch_size=2048, n_init="auto")
            ys.append(km.fit_predict(X))
        run0 = ys[0]
        aris = [adjusted_rand_score(run0, ys[i]) for i in range(1, len(ys))]
        stability["ari_runs"] = aris
        stability["ari_mean_vs_run0"] = float(np.mean(aris)) if aris else np.nan

    rows = []
    for cname in cluster_names:
        cid = name_to_id[cname]
        n = int((y_id == cid).sum())
        terms = top_terms_for_cluster(X, y_id, feat, cid, top_n=20)
        coh = mean_intra_cluster_cosine(X, y_id, cid, sample_cap=200)

        # Artifact fractions
        art_cols = [c for c in df.columns if c.startswith("art_")]
        sub = df[y_id == cid]
        art_any = float(sub[art_cols].any(axis=1).mean()) if len(sub) else np.nan

        rows.append({
            "cluster": cname,
            "n": n,
            "mean_intra_cosine": coh,
            "artifact_any_frac": art_any,
            "top_terms": ", ".join(terms),
        })

    report = pd.DataFrame(rows).sort_values(["n"], ascending=False)

    # Merge suggestions: high Jaccard overlap of top terms
    top_terms_map = {r["cluster"]: [t.strip() for t in r["top_terms"].split(",") if t.strip()] for _, r in report.iterrows()}
    merges = []
    clusters = list(top_terms_map.keys())
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            a, b = clusters[i], clusters[j]
            jac = jaccard(top_terms_map[a], top_terms_map[b])
            if jac >= 0.35:  # tweakable
                merges.append({"cluster_a": a, "cluster_b": b, "jaccard_top_terms": jac})
    merges_df = pd.DataFrame(merges).sort_values("jaccard_top_terms", ascending=False) if merges else pd.DataFrame(
        columns=["cluster_a", "cluster_b", "jaccard_top_terms"]
    )

    # Save
    import os
    os.makedirs(args.outdir, exist_ok=True)
    report_path = os.path.join(args.outdir, "cluster_validation_report.csv")
    merges_path = os.path.join(args.outdir, "cluster_merge_suggestions.csv")

    report.to_csv(report_path, index=False)
    merges_df.to_csv(merges_path, index=False)

    print(f"[OK] Rows analyzed: {len(df):,} clusters={K}")
    print(f"[OK] Stability ARI mean vs run0: {stability['ari_mean_vs_run0']}")
    print(f"[Saved] {report_path}")
    print(f"[Saved] {merges_path}")

if __name__ == "__main__":
    main()
