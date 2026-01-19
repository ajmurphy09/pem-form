#!/usr/bin/env python3
"""
Extract r/CFS submissions + comments from Pushshift-style .zst NDJSON dumps
and produce a compact, privacy-minimizing dataset with PEM detection.

Usage examples:
  python3 extract_cfs_pem.py \
    --submissions RS_CFS.zst \
    --comments RC_CFS.zst \
    --out cfs_clean.parquet

If you only have one file, you can pass just --comments or just --submissions.
"""

import argparse
import io
import json
import re
import sys
from datetime import datetime, timezone

try:
    import zstandard as zstd
except ImportError:
    print("Missing dependency: zstandard. Install with: pip install zstandard", file=sys.stderr)
    raise

# Parquet is optional; CSV fallback is always available
PARQUET_OK = True
try:
    import pandas as pd
except ImportError:
    PARQUET_OK = False

PEM_REGEX = re.compile(
    r"\b("
    r"pem|post[-\s]?exertional|post[-\s]?exercise|post[-\s]?exertion"
    r"|exertion(?:al)?\s+(?:intolerance|crash|malaise)"
    r"|crash(?:ing|ed|es)?|payback|boom\s*and\s*bust"
    r"|delayed\s+(?:crash|reaction|symptoms)"
    r")\b",
    re.IGNORECASE
)

def iter_zst_ndjson(path: str):
    """Yield dict objects from a .zst compressed NDJSON file streaming."""
    with open(path, "rb") as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Rare corrupted line; skip
                    continue

def norm_ts(created_utc):
    try:
        # created_utc is typically epoch seconds
        dt = datetime.fromtimestamp(int(created_utc), tz=timezone.utc)
        return dt.isoformat()
    except Exception:
        return None

def clean_record(obj: dict, kind: str):
    """
    kind: 'submission' or 'comment'
    Returns a compact dict or None
    """
    subreddit = (obj.get("subreddit") or "").lower()
    if subreddit != "cfs":
        return None

    created = norm_ts(obj.get("created_utc"))
    score = obj.get("score")

    if kind == "submission":
        title = obj.get("title") or ""
        selftext = obj.get("selftext") or ""
        text = (title + "\n\n" + selftext).strip()
        # Helpful for thread-level grouping later (no author stored)
        link_id = obj.get("id")
        parent_id = None
    else:
        text = (obj.get("body") or "").strip()
        link_id = obj.get("link_id")
        parent_id = obj.get("parent_id")

    # Force stable types for parquet (some dumps have ints here)
    if link_id is not None and not isinstance(link_id, str):
        link_id = str(link_id)
    if parent_id is not None and not isinstance(parent_id, str):
        parent_id = str(parent_id)

    if not text:
        return None

    pem_flag = bool(PEM_REGEX.search(text))

    return {
        "kind": kind,
        "created_utc": created,
        "score": score,
        "link_id": link_id,
        "parent_id": parent_id,
        "text": text,
        "pem_related": pem_flag,
        "text_len": len(text),
    }

def collect(path: str, kind: str, limit: int | None):
    rows = []
    n_in = 0
    n_keep = 0
    for obj in iter_zst_ndjson(path):
        n_in += 1
        rec = clean_record(obj, kind)
        if rec is not None:
            rows.append(rec)
            n_keep += 1
        if limit and n_keep >= limit:
            break
        if n_in % 500000 == 0:
            print(f"[{kind}] scanned={n_in:,} kept={n_keep:,}", file=sys.stderr)
    print(f"[{kind}] DONE scanned={n_in:,} kept={n_keep:,}", file=sys.stderr)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submissions", help="Path to subreddit submissions .zst (RS_...)", default=None)
    ap.add_argument("--comments", help="Path to subreddit comments .zst (RC_...)", default=None)
    ap.add_argument("--out", help="Output file (.parquet or .csv)", required=True)
    ap.add_argument("--limit", type=int, default=None, help="Stop after keeping N records (debug)")
    args = ap.parse_args()

    all_rows = []

    if not args.submissions and not args.comments:
        ap.error("Provide --submissions and/or --comments")

    if args.submissions:
        all_rows.extend(collect(args.submissions, "submission", args.limit))
    if args.comments:
        all_rows.extend(collect(args.comments, "comment", args.limit))

    # Sort by time for convenience
    all_rows.sort(key=lambda r: (r["created_utc"] or "", r["kind"]))

    out = args.out.lower()
    if out.endswith(".parquet"):
        if not PARQUET_OK:
            print("pandas not installed; cannot write parquet. Install: pip install pandas pyarrow", file=sys.stderr)
            sys.exit(2)
        df = pd.DataFrame(all_rows)
        # pyarrow engine preferred
        df.to_parquet(args.out, index=False)
        print(f"[OK] Wrote {args.out} rows={len(df):,}")
    elif out.endswith(".csv"):
        import csv
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()) if all_rows else [])
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"[OK] Wrote {args.out} rows={len(all_rows):,}")
    else:
        print("Output must end with .parquet or .csv", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()

