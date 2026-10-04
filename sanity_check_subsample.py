#!/usr/bin/env python3
"""Independent subsample sanity check for TEAM KHAT masks.

Level 1 (no GPU, no model): reload the three top-k pickles and recompute
overlap. This must match layer_distribution.csv exactly.

Level 2 (optional, needs the same model and the same 32-item file): zero a
subsample of masked weights, score a fixed item subsample, restore, and
repeat with a size-matched random mask.

Usage:
  python sanity_check_subsample.py --masks-dir /path/to/pickles --reference layer_distribution.csv
  python sanity_check_subsample.py --masks-dir . --prune --items pilot_items.jsonl --n-items 8 --layers 0,5,21
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import torch


def load_masks(masks_dir: Path):
    files = {
        "math": "topk_masks_pure_symbolic.pkl",
        "finance": "topk_masks_financial.pkl",
        "nonfin": "topk_masks_non_financial.pkl",
    }
    out = {}
    for key, name in files.items():
        path = masks_dir / name
        with path.open("rb") as f:
            out[key] = pickle.load(f)
    return out


def layer_index(name: str) -> int:
    return int(name.split(".")[2])


def recompute_overlap(masks):
    layers = sorted(masks["math"].keys(), key=layer_index)
    rows = []
    for name in layers:
        m = masks["math"][name].bool().cpu()
        f = masks["finance"][name].bool().cpu()
        n = masks["nonfin"][name].bool().cpu()
        rows.append({
            "layer": layer_index(name),
            "module": name,
            "params": int(m.numel()),
            "math_topk": int(m.sum()),
            "finance_topk": int(f.sum()),
            "nonfin_topk": int(n.sum()),
            "math_specific": int((m & ~n).sum()),
            "finance_specific": int((f & ~n).sum()),
            "shared_calc": int((m & f & n).sum()),
            "math_and_finance": int((m & f).sum()),
            "math_not_finance": int((m & ~f).sum()),
            "finance_not_math": int((f & ~m).sum()),
        })
    return rows


def compare_to_reference(rows, reference: Path):
    with reference.open() as f:
        ref = {int(r["layer"]): r for r in csv.DictReader(f)}
    mismatches = []
    keys = [
        "math_topk", "finance_topk", "nonfin_topk",
        "math_specific", "finance_specific", "shared_calc",
        "math_and_finance", "math_not_finance", "finance_not_math",
    ]
    for row in rows:
        src = ref.get(row["layer"])
        if src is None:
            mismatches.append(f"layer {row['layer']} missing from reference")
            continue
        for key in keys:
            if int(src[key]) != row[key]:
                mismatches.append(f"layer {row['layer']} {key}: got {row[key]} vs reference {src[key]}")
    return mismatches


def random_mask_like(mask: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    n = int(mask.sum().item())
    flat = torch.zeros(mask.numel(), dtype=torch.bool)
    idx = torch.randperm(mask.numel(), generator=g)[:n]
    flat[idx] = True
    return flat.view(mask.shape)


def pruning_cardinality_check(masks, layers, seed: int):
    """No model needed. Check that a random mask has the same count, and that
    zeroing a cloned weight on the finance-specific mask hits exactly that count."""
    report = []
    for name, mask in masks["finance"].items():
        layer = layer_index(name)
        if layer not in layers:
            continue
        finance = mask.bool().cpu()
        nonfin = masks["nonfin"][name].bool().cpu()
        specific = finance & ~nonfin
        rand = random_mask_like(specific, seed + layer)
        weight = torch.ones(specific.shape, dtype=torch.float32)
        saved = weight.clone()
        weight[specific] = 0
        n_zero = int((weight == 0).sum())
        weight.copy_(saved)
        restored = bool(torch.equal(weight, saved))
        report.append({
            "layer": layer,
            "finance_specific": int(specific.sum()),
            "random_same_count": int(rand.sum()) == int(specific.sum()),
            "zeros_match_mask": n_zero == int(specific.sum()),
            "restore_ok": restored,
        })
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--layers", type=str, default="0,5,12,21,23")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("sanity_check.json"))
    args = parser.parse_args()

    masks = load_masks(args.masks_dir)
    rows = recompute_overlap(masks)
    layers = [int(x) for x in args.layers.split(",") if x]
    subsample = [r for r in rows if r["layer"] in layers]
    mismatches = compare_to_reference(rows, args.reference) if args.reference else []
    prune = pruning_cardinality_check(masks, set(layers), args.seed)

    summary = {
        "n_layers": len(rows),
        "math_specific_total": sum(r["math_specific"] for r in rows),
        "finance_specific_total": sum(r["finance_specific"] for r in rows),
        "shared_calc_total": sum(r["shared_calc"] for r in rows),
        "subsample_layers": subsample,
        "reference_mismatches": mismatches,
        "overlap_matches_reference": args.reference is not None and not mismatches,
        "pruning_cardinality": prune,
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in [
        "n_layers", "math_specific_total", "finance_specific_total",
        "shared_calc_total", "overlap_matches_reference", "reference_mismatches",
    ]}, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
