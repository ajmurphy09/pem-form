python3 - <<'EOF'
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

INP = "~/Downloads/cfs_pem_enriched.parquet"
OUT = "~/Downloads/cfs_pem_symptom_clustered_sample.parquet"

MODE = "submissions_only"
MAX_ROWS = 60000
RANDOM_STATE = 42

K_MIN, K_MAX = 8, 24
TOP_TERMS = 18
EXAMPLES_PER_CLUSTER = 4

URL_RE = re.compile(r"https?://\S+|www\.\S+")
WS_RE = re.compile(r"\s+")

# Heuristic “context” removals (we keep it conservative)
CONTEXT_PAT = re.compile(
    r"\b(doctor|doctors|diagnos|test|blood|mri|clinic|specialist|gp|insurance|disab|work|job|boss|hr dept|school|college|class|university|family|mom|dad|husband|wife|boyfriend|girlfriend|relationship|friend|reddit|subreddit|lurking|posting|commenting)\b",
    re.I
)

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\u200b", " ")
    s = s.replace("&amp;", "and").replace("&nbsp;", " ").replace("x200b", " ")
    s = URL_RE.sub(" ", s)
    s = s.lower()
    s = WS_RE.sub(" ", s).strip()
    return s

def symptom_only(s: str) -> str:
    # Split into sentences and drop ones that are mostly context
    parts = re.split(r"(?<=[\.\?!])\s+", s)
    kept = []
    for p in parts:
        p = p.strip()
        if len(p) < 20:
            continue
        # if sentence hits lots of context words, drop it
        if CONTEXT_PAT.search(p):
            continue
        kept.append(p)
    # If we dropped everything, fall back to the full text
    return " ".join(kept) if kept else s

df = pd.read_parquet(INP)
df = df[df["pem_related"] == True].copy()
if MODE == "submissions_only":
    df = df[df["kind"] == "submission"].copy()

if len(df) > MAX_ROWS:
    df = df.sample(MAX_ROWS, random_state=RANDOM_STATE).copy()

df["clean"] = df["text"].map(clean_text).map(symptom_only)
df = df[df["clean"].str.len() >= 120].copy()
df = df.reset_index(drop=True)

print(f"Rows for symptom-only clustering: {len(df):,} (mode={MODE})")

vectorizer = TfidfVectorizer(
    max_features=120000,
    ngram_range=(1,3),
    min_df=6,
    max_df=0.30,
    stop_words="english",
)
X = vectorizer.fit_transform(df["clean"].tolist())
print("TF-IDF matrix:", X.shape)

score_sample_n = min(15000, X.shape[0])
rng = np.random.default_rng(RANDOM_STATE)
sample_idx = rng.choice(X.shape[0], size=score_sample_n, replace=False)
X_sample = X[sample_idx]

best = None
for k in range(K_MIN, K_MAX + 1):
    km = KMeans(n_clusters=k, n_init="auto", random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    s = silhouette_score(X_sample, labels[sample_idx], metric="cosine")
    print(f"K={k:2d}  silhouette={s:.4f}")
    if best is None or s > best[1]:
        best = (k, s, km, labels)

best_k, best_s, km, labels = best
print("\nBest K:", best_k, "silhouette:", round(best_s, 4))

df["cluster"] = labels

feature_names = np.array(vectorizer.get_feature_names_out())
centers = km.cluster_centers_

def top_terms_for_cluster(c, n=TOP_TERMS):
    row = centers[c]
    top_idx = np.argsort(row)[::-1][:n]
    return feature_names[top_idx].tolist()

counts = df["cluster"].value_counts().sort_index()
print("\nCluster sizes:")
for c, n in counts.items():
    print(f"  C{c:02d}: {n:,}")

print("\nTop terms per cluster:")
for c in range(best_k):
    terms = top_terms_for_cluster(c)
    print(f"\nC{c:02d}  (n={counts.get(c,0):,})")
    print("  " + ", ".join(terms))

print("\n\nEXEMPLARS (first 320 chars):")
for c in range(best_k):
    idxs = df.index[df["cluster"] == c].to_numpy()
    if len(idxs) == 0:
        continue
    centroid = centers[c].reshape(1, -1)
    sims = cosine_similarity(X[idxs], centroid).ravel()
    top_local = np.argsort(sims)[::-1][:EXAMPLES_PER_CLUSTER]
    chosen = idxs[top_local]
    print(f"\n=== C{c:02d} (n={counts.get(c,0):,}) ===")
    for ix in chosen:
        t = re.sub(r"\s+", " ", df.loc[ix, "text"]).strip()
        print("-", t[:320])

df.drop(columns=["clean"]).to_parquet(OUT, index=False)
print("\nSaved:", OUT)
EOF
