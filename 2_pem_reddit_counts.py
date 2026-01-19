python3 - <<'EOF'
import pandas as pd

df = pd.read_parquet('~/Downloads/cfs_clean.parquet')

print("Total rows:", len(df))
print("PEM-related rows:", df['pem_related'].sum())
print("Percent PEM:", round(df['pem_related'].mean() * 100, 2), "%")

# Show a few PEM examples
pem = df[df['pem_related']].sample(5, random_state=42)
for i, row in pem.iterrows():
    print("\n---")
    print(row['text'][:500])
EOF
