python3 - <<'EOF'
import os
import pandas as pd

INP = os.path.expanduser("~/Downloads/cfs_pem_symptom_clustered_sample.parquet")
OUT_DIR = os.path.expanduser("~/Downloads/cfs_pem_reports")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_parquet(INP)
print("Loaded:", INP)
print("Rows:", f"{len(df):,}")
print("Columns:", ", ".join(df.columns[:30]) + (" ..." if len(df.columns) > 30 else ""))

# ---- sanity: required fields ----
required_any = ["cluster", "text"]
for c in required_any:
    if c not in df.columns:
        raise SystemExit(f"Missing required column: {c}")

# These are what we want to cross-tab if present
meta_cols = ["pem_onset", "pem_duration", "pem_trigger", "kind"]
present_meta = [c for c in meta_cols if c in df.columns]
missing_meta = [c for c in meta_cols if c not in df.columns]

if missing_meta:
    print("\n[NOTE] Missing columns (we'll still do prevalence):", ", ".join(missing_meta))
if present_meta:
    print("[OK] Meta columns present:", ", ".join(present_meta))

# ---- Define phenotype labels (edit freely) ----
# Based on your cluster output:
PHENOTYPE_MAP = {
    2:  "immune_flu_like",        # sore throat / lymph / fever / chills
    12: "autonomic_hr_pots",      # HR threshold / POTS / standing / bpm
    17: "thermoregulation_temp",  # cold/hot/shower/weather triggers
    6:  "neurocognitive_brainfog",
    9:  "pain_dominant",
    16: "sleep_dominant",
    0:  "hyperadrenergic_wired",  # adrenaline / wired-but-tired
    22: "lactate_burning",        # lactic acid / burning muscles
}

df["phenotype"] = df["cluster"].map(PHENOTYPE_MAP).fillna("other_or_context")

# ---- 1) Prevalence ----
prev = (
    df["phenotype"]
    .value_counts(dropna=False)
    .rename_axis("phenotype")
    .reset_index(name="rows")
)
prev["percent"] = (prev["rows"] / len(df) * 100).round(2)

# Put phenotype groups first (in a sensible order), then other
order = [
    "immune_flu_like",
    "autonomic_hr_pots",
    "thermoregulation_temp",
    "neurocognitive_brainfog",
    "pain_dominant",
    "sleep_dominant",
    "hyperadrenergic_wired",
    "lactate_burning",
    "other_or_context",
]
prev["phenotype"] = pd.Categorical(prev["phenotype"], categories=order, ordered=True)
prev = prev.sort_values(["phenotype"]).reset_index(drop=True)

prev_path = os.path.join(OUT_DIR, "prevalence_by_phenotype.csv")
prev.to_csv(prev_path, index=False)

print("\n=== Prevalence by phenotype ===")
print(prev.to_string(index=False))

# ---- helper: cross-tab with % within phenotype ----
def crosstab_with_pct(df, row_col, col_col, out_name):
    tab = pd.crosstab(df[row_col], df[col_col], dropna=False)
    pct = tab.div(tab.sum(axis=1), axis=0) * 100
    pct = pct.round(2)

    # Save both counts and pct
    tab_path = os.path.join(OUT_DIR, f"{out_name}_counts.csv")
    pct_path = os.path.join(OUT_DIR, f"{out_name}_pct_within_{row_col}.csv")
    tab.to_csv(tab_path)
    pct.to_csv(pct_path)

    print(f"\n=== {out_name}: counts ===")
    print(tab.to_string())
    print(f"\n=== {out_name}: % within {row_col} ===")
    print(pct.to_string())

# ---- 2) Cross-tabs (only if columns exist) ----
if "pem_trigger" in df.columns:
    crosstab_with_pct(df, "phenotype", "pem_trigger", "phenotype_x_trigger")

if "pem_onset" in df.columns:
    crosstab_with_pct(df, "phenotype", "pem_onset", "phenotype_x_onset")

if "pem_duration" in df.columns:
    crosstab_with_pct(df, "phenotype", "pem_duration", "phenotype_x_duration")

if "kind" in df.columns:
    crosstab_with_pct(df, "phenotype", "kind", "phenotype_x_kind")

# ---- 3) Save a labeled parquet for easy browsing ----
labeled_path = os.path.join(OUT_DIR, "cfs_pem_symptom_labeled.parquet")
df.to_parquet(labeled_path, index=False)

print("\nSaved files to:", OUT_DIR)
print(" -", prev_path)
print(" -", labeled_path)
print("Plus any cross-tab CSVs that were possible based on available columns.")
EOF
