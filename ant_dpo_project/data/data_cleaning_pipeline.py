"""Preference-data cleaning pipeline.

A staged filter that takes raw ``(prompt, chosen, rejected)`` pairs and removes
the junk that would otherwise poison DPO:

    1. exact dedup            (identical chosen/rejected text)
    2. near-dedup (SimHash)   (boilerplate templated responses)
    3. length filter          (degenerate too-short / runaway too-long)
    4. language consistency    (non-English leakage)
    5. toxicity filter         (keyword screen; a real run would use Detoxify)

Each stage records how many rows it drops so we can emit a retention report. The
SimHash implementation is dependency-free (stdlib ``hashlib``) so the pipeline
runs anywhere; the language/toxicity stages are heuristic by default and document
where a production system would swap in ``fasttext`` / ``Detoxify``.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass, field

import pandas as pd


# --------------------------------------------------------------------------- #
# SimHash (64-bit) for near-duplicate detection                                #
# --------------------------------------------------------------------------- #
def _tokens(text: str, k: int = 3):
    """Word k-shingles."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < k:
        return [" ".join(words)] if words else []
    return [" ".join(words[i:i + k]) for i in range(len(words) - k + 1)]


def simhash(text: str, bits: int = 64) -> int:
    """Charikar SimHash: near-identical texts get near-identical fingerprints."""
    v = [0] * bits
    for sh in _tokens(text):
        h = int.from_bytes(hashlib.blake2b(sh.encode(), digest_size=8).digest(), "big")
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(bits):
        if v[i] > 0:
            out |= (1 << i)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def near_duplicate_mask(texts, threshold: int = 3, band_bits: int = 16):
    """Return a boolean keep-mask, dropping near-duplicates (Hamming <= threshold).

    Uses LSH banding on SimHash prefixes to avoid the O(n^2) all-pairs scan: only
    fingerprints sharing a band are compared. The first item in each near-dup
    cluster is kept.
    """
    hashes = [simhash(t) for t in texts]
    keep = [True] * len(texts)
    buckets: dict = {}
    mask_lo = (1 << band_bits) - 1
    n_bands = 64 // band_bits
    for i, h in enumerate(hashes):
        if not keep[i]:
            continue
        candidates = set()
        for b in range(n_bands):
            band_key = (b, (h >> (b * band_bits)) & mask_lo)
            candidates.update(buckets.get(band_key, ()))
        for j in candidates:
            if keep[j] and hamming(hashes[i], hashes[j]) <= threshold:
                keep[i] = False
                break
        if keep[i]:
            for b in range(n_bands):
                band_key = (b, (h >> (b * band_bits)) & mask_lo)
                buckets.setdefault(band_key, []).append(i)
    return keep


# --------------------------------------------------------------------------- #
# heuristic language / toxicity screens                                        #
# --------------------------------------------------------------------------- #
_NON_ASCII = re.compile(r"[^\x00-\x7F]")
_TOXIC_WORDS = {"idiot", "stupid", "dumb", "hate", "trash", "moron"}


def is_english(text: str, max_non_ascii_frac: float = 0.05) -> bool:
    if not text:
        return False
    non_ascii = len(_NON_ASCII.findall(text))
    return (non_ascii / max(1, len(text))) <= max_non_ascii_frac


def is_toxic(text: str) -> bool:
    words = set(re.findall(r"\w+", text.lower()))
    return bool(words & _TOXIC_WORDS)


# --------------------------------------------------------------------------- #
# pipeline                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class CleaningReport:
    stages: list = field(default_factory=list)  # (name, n_before, n_after, dropped)

    def add(self, name, before, after):
        self.stages.append((name, before, after, before - after))

    @property
    def initial(self):
        return self.stages[0][1] if self.stages else 0

    @property
    def final(self):
        return self.stages[-1][2] if self.stages else 0

    def to_markdown(self) -> str:
        lines = ["## Data cleaning report", "",
                 "| stage | in | out | dropped | drop % |",
                 "|---|---|---|---|---|"]
        for name, before, after, dropped in self.stages:
            pct = 100 * dropped / max(1, before)
            lines.append(f"| {name} | {before} | {after} | {dropped} | {pct:.1f}% |")
        overall = 100 * self.final / max(1, self.initial)
        lines += ["", f"**Overall retention: {self.final}/{self.initial} "
                  f"= {overall:.1f}%**"]
        return "\n".join(lines)


def clean_pairs(
    pairs: pd.DataFrame,
    min_len: int = 8,
    max_len: int = 2000,
    simhash_threshold: int = 3,
    report: CleaningReport | None = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    report = report or CleaningReport()
    df = pairs.copy()
    report.add("input", len(df), len(df))

    # 1. exact dedup on (chosen, rejected)
    before = len(df)
    df = df.drop_duplicates(subset=["chosen", "rejected"]).reset_index(drop=True)
    report.add("exact_dedup", before, len(df))

    # 2. near-dedup via SimHash over the chosen text
    before = len(df)
    keep = near_duplicate_mask(list(df["chosen"]), threshold=simhash_threshold)
    df = df[keep].reset_index(drop=True)
    report.add("near_dedup_simhash", before, len(df))

    # 3. length filter (both sides must be within bounds)
    before = len(df)
    def _ok_len(s):
        return df[s].str.len().between(min_len, max_len)
    df = df[_ok_len("chosen") & _ok_len("rejected")].reset_index(drop=True)
    report.add("length_filter", before, len(df))

    # 4. language consistency
    before = len(df)
    m = df["chosen"].map(is_english) & df["rejected"].map(is_english)
    df = df[m].reset_index(drop=True)
    report.add("language_filter", before, len(df))

    # 5. toxicity screen
    before = len(df)
    m = ~(df["chosen"].map(is_toxic) | df["rejected"].map(is_toxic))
    df = df[m].reset_index(drop=True)
    report.add("toxicity_filter", before, len(df))

    return df, report


def main():
    ap = argparse.ArgumentParser(description="Clean preference pairs")
    ap.add_argument("--in", dest="inp", default="results/preference_pairs.parquet")
    ap.add_argument("--out", default="results/preference_pairs_clean.parquet")
    ap.add_argument("--report", default="results/data_cleaning_report.md")
    args = ap.parse_args()

    pairs = (pd.read_parquet(args.inp) if args.inp.endswith(".parquet")
             else pd.read_csv(args.inp))
    clean, report = clean_pairs(pairs)
    print(report.to_markdown())
    with open(args.report, "w") as f:
        f.write(report.to_markdown() + "\n")
    try:
        clean.to_parquet(args.out)
    except Exception:
        args.out = args.out.replace(".parquet", ".csv")
        clean.to_csv(args.out, index=False)
    print(f"\n[clean] wrote {args.out} and {args.report}")


if __name__ == "__main__":
    main()
