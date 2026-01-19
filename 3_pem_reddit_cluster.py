python3 - <<'EOF'
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

INP = "~/Downloads/cfs_pem_enriched.parquet"
MODE = "submissions_only"   # "submissions_only" or "all_pem"
MAX_ROWS = 60000
RANDOM_STATE = 42

K_MIN, K_MAX = 8, 24
TOP_TERMS = 18
EXAMPLES_PER_CLUSTER = 4

df0 = pd.read_parquet(INP)
df0 = df0[df0["pem_related"] == True].copy()
if MODE == "submissions_only":
    df0 = df0[df0["kind"] == "submission"].copy()

if len(df0) > MAX_ROWS:
    df0 = df0.sample(MAX_ROWS, random_state=RANDOM_STATE).copy()

# Light clean
URL_RE = re.compile(r"https?://\S+|www\.\S+")
WS_RE = re.compile(r"\s+")

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\u200b", " ")
    s = URL_RE.sub(" ", s)
    s = s.lower()
    s = WS_RE.sub(" ", s).strip()
    return s

df0["clean"] = df0["text"].map(clean_text)

# Drop short texts
df0 = df0[df0["clean"].str.len() >= 120].copy()

# Reset index to avoid sparse indexing mismatches
df0 = df0.reset_index(drop=True)

print(f"Rows for clustering: {len(df0):,} (mode={MODE})")

vectorizer = TfidfVectorizer(
    max_features=120000,
    ngram_range=(1,3),
    min_df=8,
    max_df=0.35,
    stop_words="english",
)
X = vectorizer.fit_transform(df0["clean"].tolist())
print("TF-IDF matrix:", X.shape)

# silhouette on sample for speed
score_sample_n = min(15000, X.shape[0])
rng = np.random.default_rng(RANDOM_STATE)
sample_idx = rng.choice(X.shape[0], size=score_sample_n, replace=False)
X_sample = X[sample_idx]

best = None
scores = []
for k in range(K_MIN, K_MAX + 1):
    km = KMeans(n_clusters=k, n_init="auto", random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    s = silhouette_score(X_sample, labels[sample_idx], metric="cosine")
    print(f"K={k:2d}  silhouette={s:.4f}")
    if best is None or s > best[1]:
        best = (k, s, km, labels)

best_k, best_s, km, labels = best
print("\nBest K:", best_k, "silhouette:", round(best_s, 4))

df0["cluster"] = labels

feature_names = np.array(vectorizer.get_feature_names_out())
centers = km.cluster_centers_

def top_terms_for_cluster(c, n=TOP_TERMS):
    row = centers[c]
    top_idx = np.argsort(row)[::-1][:n]
    return feature_names[top_idx].tolist()

counts = df0["cluster"].value_counts().sort_index()
print("\nCluster sizes:")
for c, n in counts.items():
    print(f"  C{c:02d}: {n:,}")

print("\nTop terms per cluster:")
for c in range(best_k):
    terms = top_terms_for_cluster(c)
    print(f"\nC{c:02d}  (n={counts.get(c,0):,})")
    print("  " + ", ".join(terms))

# Exemplars by similarity to centroid
print("\n\nEXEMPLARS (first 320 chars):")
for c in range(best_k):
    idxs = df0.index[df0["cluster"] == c].to_numpy()
    if len(idxs) == 0:
        continue
    # compute cosine similarity of each doc in cluster to its centroid vector
    centroid = centers[c].reshape(1, -1)
    # get dense similarity efficiently by selecting those rows
    sims = cosine_similarity(X[idxs], centroid).ravel()
    top_local = np.argsort(sims)[::-1][:EXAMPLES_PER_CLUSTER]
    chosen = idxs[top_local]

    print(f"\n=== C{c:02d} (n={counts.get(c,0):,}) ===")
    for ix in chosen:
        t = re.sub(r"\s+", " ", df0.loc[ix, "text"]).strip()
        print("-", t[:320])

out = "~/Downloads/cfs_pem_clustered_sample.parquet"
df0.drop(columns=["clean"]).to_parquet(out, index=False)
print("\nSaved clustered sample to:", out)
EOF
