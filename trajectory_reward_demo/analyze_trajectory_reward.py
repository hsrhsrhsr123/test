# -*- coding: utf-8 -*-
"""
Module 3: 对比分析 — 传统 pass/fail reward vs 多维度 TRM reward
================================================================

运行:
    python analyze_trajectory_reward.py

输出三部分:
  1. 全量对比表: 每条轨迹的传统 reward、TRM 各维度分、TRM 总分
  2. 分析一: 高效 vs 冗余 — 传统 reward 下两者同为 1.0 无法区分, TRM 能拉开差距
  3. 分析二: 有价值的失败 vs 垃圾失败 — 传统 reward 下两者同为 0.0,
     TRM 通过 localization / patch overlap 给有价值的失败部分分
"""

from trajectory_data import get_trajectories
from trajectory_reward import TrajectoryRewardModel, traditional_reward

CATEGORY_ZH = {
    "efficient": "高效",
    "redundant": "冗余",
    "valuable_fail": "有价值失败",
    "garbage": "垃圾",
}

DIM_COLS = [
    ("pass_rate", "pass"),
    ("step_efficiency", "eff"),
    ("error_recovery", "recov"),
    ("localization_accuracy", "loc"),
    ("patch_quality", "patch"),
]


def format_table(headers, rows):
    """简易等宽 ASCII 表格 (仅依赖标准库)。"""
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    sep = "-+-".join("-" * w for w in widths)
    out = [" | ".join(str(h).ljust(w) for h, w in zip(headers, widths)), sep]
    for r in rows:
        out.append(" | ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    return "\n".join(out)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    trm = TrajectoryRewardModel()
    results = []
    for traj in get_trajectories():
        dims = trm.score(traj)
        results.append({
            "task_id": traj["task_id"],
            "category": traj["category"],
            "n_steps": len(traj["steps"]),
            "traditional": traditional_reward(traj),
            "dims": dims,
        })

    # ------------------------------------------------------------------
    # Part 1: 全量对比表
    # ------------------------------------------------------------------
    print("=" * 100)
    print("Part 1. 全量对比: 传统 pass/fail reward vs 多维度 TRM reward")
    print("=" * 100)
    headers = (["task_id", "类别", "步数", "传统reward"]
               + [short for _, short in DIM_COLS] + ["TRM总分"])
    rows = []
    for r in results:
        rows.append([
            r["task_id"],
            CATEGORY_ZH[r["category"]],
            r["n_steps"],
            f"{r['traditional']:.1f}",
            *[f"{r['dims'][key]:.2f}" for key, _ in DIM_COLS],
            f"{r['dims']['total']:.3f}",
        ])
    print(format_table(headers, rows))

    by_cat = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    # ------------------------------------------------------------------
    # Part 2: 高效 vs 冗余 (传统 reward 均为 1.0)
    # ------------------------------------------------------------------
    print()
    print("=" * 100)
    print("Part 2. 高效轨迹 vs 冗余轨迹 —— 传统 reward 无法区分, TRM 可以")
    print("=" * 100)
    eff_trad = mean([r["traditional"] for r in by_cat["efficient"]])
    red_trad = mean([r["traditional"] for r in by_cat["redundant"]])
    eff_trm = mean([r["dims"]["total"] for r in by_cat["efficient"]])
    red_trm = mean([r["dims"]["total"] for r in by_cat["redundant"]])
    eff_steps = mean([r["n_steps"] for r in by_cat["efficient"]])
    red_steps = mean([r["n_steps"] for r in by_cat["redundant"]])
    print(format_table(
        ["类别", "平均步数", "传统reward均值", "TRM均值", "平均step_efficiency"],
        [["高效", f"{eff_steps:.1f}", f"{eff_trad:.1f}", f"{eff_trm:.3f}",
          f"{mean([r['dims']['step_efficiency'] for r in by_cat['efficient']]):.2f}"],
         ["冗余", f"{red_steps:.1f}", f"{red_trad:.1f}", f"{red_trm:.3f}",
          f"{mean([r['dims']['step_efficiency'] for r in by_cat['redundant']]):.2f}"]],
    ))
    print(f"""
结论: 两类轨迹最终都 pass, 传统 reward 都是 {eff_trad:.1f}, 训练时完全无法区分
"{eff_steps:.0f} 步直达修复" 和 "{red_steps:.0f} 步反复试错" 的轨迹质量;
TRM 通过 step_efficiency / error_recovery 维度将两者拉开 {eff_trm - red_trm:.3f}
(高效 {eff_trm:.3f} vs 冗余 {red_trm:.3f}), 为 RL 提供了偏好高效行为的梯度信号。""")

    # ------------------------------------------------------------------
    # Part 3: 有价值的失败 vs 垃圾失败 (传统 reward 均为 0.0)
    # ------------------------------------------------------------------
    print("=" * 100)
    print("Part 3. 有价值的失败 vs 垃圾轨迹 —— 传统 reward 一律给 0, TRM 给部分分")
    print("=" * 100)
    vf, gb = by_cat["valuable_fail"], by_cat["garbage"]
    print(format_table(
        ["类别", "传统reward均值", "TRM均值", "平均localization", "平均patch_quality"],
        [["有价值失败", f"{mean([r['traditional'] for r in vf]):.1f}",
          f"{mean([r['dims']['total'] for r in vf]):.3f}",
          f"{mean([r['dims']['localization_accuracy'] for r in vf]):.2f}",
          f"{mean([r['dims']['patch_quality'] for r in vf]):.2f}"],
         ["垃圾", f"{mean([r['traditional'] for r in gb]):.1f}",
          f"{mean([r['dims']['total'] for r in gb]):.3f}",
          f"{mean([r['dims']['localization_accuracy'] for r in gb]):.2f}",
          f"{mean([r['dims']['patch_quality'] for r in gb]):.2f}"]],
    ))
    vf_trm = mean([r["dims"]["total"] for r in vf])
    gb_trm = mean([r["dims"]["total"] for r in gb])
    print(f"""
结论: 两类轨迹最终都 fail, 传统 reward 一律给 0, "正确定位到 bug 但修复方向错误"
的轨迹与"完全没理解问题"的轨迹在训练中被同等惩罚, 浪费了宝贵的探索信号;
TRM 通过 localization_accuracy (定位到正确文件/函数) 和 patch_quality (与
ground truth 的部分 overlap) 给前者 {vf_trm:.3f} 的部分分 (垃圾轨迹仅 {gb_trm:.3f}),
使 RL 能从失败轨迹中学到 "定位是对的, 只是修复策略需要改进"。""")

    # ------------------------------------------------------------------
    # 总结
    # ------------------------------------------------------------------
    print("=" * 100)
    print("总结: 传统 reward 只有 {0, 1} 两个取值, 8 条轨迹被压缩成 2 个等价类;")
    trm_scores = sorted((r["dims"]["total"] for r in results), reverse=True)
    print(f"TRM 给出 8 个可区分的细粒度分数: "
          f"{', '.join(f'{s:.3f}' for s in trm_scores)}")
    print("细粒度信号 = 更低方差的 advantage 估计 + 可从失败中学习 => 更高效的 Agentic RL。")
    print("=" * 100)


if __name__ == "__main__":
    main()
