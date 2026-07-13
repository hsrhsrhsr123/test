#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROPELLA-Code Step 3: 筛选策略对比分析。

对比两种数据筛选策略:
  - baseline:按总分 top-k 筛选(传统单分数过滤);
  - propella:按多属性规则筛选(默认 逻辑密度>=3 AND 自包含性>=3,不看总分)。

为公平对比,baseline 的 k 默认取 propella 选中的条数(也可 --topk 指定),
然后输出两个子集在各维度上的均值/分布差异,以及两种策略的分歧样本——
分歧样本正是"多属性标注优于单一总分"的直接证据。

用法:
    python analyze_propella.py --input annotations.jsonl
    python analyze_propella.py --input annotations.jsonl \\
        --rule "logic_density>=3,self_containment>=3,practicality>=2" --topk 8
"""
import argparse
import json
import sys

from annotation_prompt import DIMENSIONS, DIMENSION_NAMES_ZH

_OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
}


def parse_rule(rule_str):
    """解析 'dim>=3,dim2<=4' 形式的规则,返回 [(dim, op_str, fn, 阈值)]。"""
    conds = []
    for part in rule_str.split(","):
        part = part.strip()
        if not part:
            continue
        for op in (">=", "<=", "==", ">", "<"):  # 长运算符优先匹配
            if op in part:
                dim, val = part.split(op, 1)
                dim = dim.strip()
                if dim not in DIMENSIONS:
                    sys.exit(f"未知维度: {dim},可选: {DIMENSIONS}")
                conds.append((dim, op, _OPS[op], int(val)))
                break
        else:
            sys.exit(f"无法解析条件: {part!r}")
    return conds


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def dim_stats(records):
    """返回 {dim: (均值, {分数: 条数})}。"""
    stats = {}
    for d in DIMENSIONS:
        vals = [r[d] for r in records]
        hist = {s: sum(1 for v in vals if v == s) for s in range(1, 6)}
        stats[d] = (mean(vals), hist)
    return stats


def fmt_hist(hist):
    return " ".join(f"{s}:{hist[s]}" for s in range(1, 6))


def print_table(title, records):
    print(f"\n### {title}(n={len(records)})")
    if not records:
        print("  (空集)")
        return
    stats = dim_stats(records)
    print(f"  {'维度':<14}{'均值':>6}   分布(分数:条数)")
    for d in DIMENSIONS:
        avg, hist = stats[d]
        name = DIMENSION_NAMES_ZH[d]
        print(f"  {name:<12}{avg:>6.2f}   {fmt_hist(hist)}")
    print(f"  {'总分均值':<12}{mean([r['total'] for r in records]):>6.2f}")


def print_samples(title, records, note=""):
    print(f"\n### {title}(n={len(records)}){note}")
    if not records:
        print("  (无)")
        return
    header = f"  {'id':<28}" + "".join(f"{d[:5]:>7}" for d in DIMENSIONS) + f"{'总分':>6}"
    print(header)
    for r in sorted(records, key=lambda r: -r["total"]):
        row = f"  {r['id']:<28}" + "".join(f"{r[d]:>7}" for d in DIMENSIONS)
        print(row + f"{r['total']:>6}")


def main():
    ap = argparse.ArgumentParser(description="PROPELLA-Code 筛选策略对比")
    ap.add_argument("--input", required=True, help="标注结果 jsonl")
    ap.add_argument("--rule", default="logic_density>=3,self_containment>=3",
                    help="propella 多属性规则,逗号分隔的 dim>=n 条件(AND 关系)")
    ap.add_argument("--topk", type=int, default=None,
                    help="baseline 的 k;默认与 propella 选中条数对齐")
    args = ap.parse_args()

    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                print(f"[warn] 跳过标注失败样本: {r['id']}", file=sys.stderr)
                continue
            records.append(r)
    if not records:
        sys.exit("没有可用的标注结果")

    conds = parse_rule(args.rule)

    # -------- 策略 1: propella 多属性规则(AND,不看总分)
    propella_sel = [r for r in records
                    if all(fn(r[d], v) for d, _, fn, v in conds)]

    # -------- 策略 2: baseline 总分 top-k(平分时按 id 排序保证确定性)
    k = args.topk if args.topk is not None else len(propella_sel)
    baseline_sel = sorted(records, key=lambda r: (-r["total"], r["id"]))[:k]

    p_ids = {r["id"] for r in propella_sel}
    b_ids = {r["id"] for r in baseline_sel}
    both, only_b, only_p = p_ids & b_ids, b_ids - p_ids, p_ids - b_ids

    rule_str = " AND ".join(f"{DIMENSION_NAMES_ZH[d]}{op}{v}"
                            for d, op, _, v in conds)
    print("=" * 72)
    print("PROPELLA-Code 筛选策略对比")
    print("=" * 72)
    print(f"总样本数: {len(records)}")
    print(f"baseline : 总分 top-{k}")
    print(f"propella : {rule_str}")
    print(f"重合 {len(both)} 条 | 仅 baseline {len(only_b)} 条 | "
          f"仅 propella {len(only_p)} 条 | "
          f"Jaccard = {len(both) / max(len(p_ids | b_ids), 1):.2f}")

    print_table("全量数据", records)
    print_table("baseline 选中(总分 top-k)", baseline_sel)
    print_table("propella 选中(多属性规则)", propella_sel)

    # -------- 各维度均值差:propella 相对 baseline
    if propella_sel and baseline_sel:
        ps, bs = dim_stats(propella_sel), dim_stats(baseline_sel)
        print("\n### 维度均值差(propella - baseline)")
        for d in DIMENSIONS:
            diff = ps[d][0] - bs[d][0]
            bar = "+" * int(round(abs(diff) * 4)) if diff > 0 else \
                  "-" * int(round(abs(diff) * 4))
            print(f"  {DIMENSION_NAMES_ZH[d]:<12}{diff:+6.2f}  {bar}")

    # -------- 分歧样本:两种策略"吵架"的地方,是论文故事的核心
    print_samples("仅 baseline 选中(总分高但存在短板,propella 拒绝)",
                  [r for r in records if r["id"] in only_b])
    print_samples("仅 propella 选中(总分不占优但关键属性达标,baseline 漏掉)",
                  [r for r in records if r["id"] in only_p])

    print("\n解读: '仅 baseline' 一栏通常是总分被可读性/教学性等维度抬高、"
          "\n但逻辑密度或自包含性有硬伤的样本;'仅 propella' 一栏则是"
          "\n可读性差拉低了总分、但作为预训练语料实际有价值的样本。"
          "\n两栏差异越大,说明单一总分损失的信息越多。")


if __name__ == "__main__":
    main()
