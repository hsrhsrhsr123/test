# -*- coding: utf-8 -*-
"""
Module 2: 多维度 Trajectory Reward Model (TRM)
==============================================

对 Code Agent 执行轨迹进行细粒度、多维度的质量评估, 输出 5 个维度的分数
(各 0-1) 以及加权总分, 作为 Agentic RL 训练的 reward 信号。

demo 阶段全部用规则计算, 不训练模型; 接口设计上与可学习的 reward model
对齐 (输入一条轨迹, 输出一个标量 reward + 各维度分解)。

五个维度:
  pass_rate             最终测试是否通过 (0/1)
  step_efficiency       有效步骤占比 (成功的 edit/test 为有效, 命中 bug 文件的
                        search 记半分, 其余 search/think 及出错步骤为无效)
  error_recovery        遇到 error/timeout 后是否在窗口内有效恢复
  localization_accuracy 是否正确定位到 bug 所在文件 (0.6) 与函数/类 (0.4)
  patch_quality         最终 patch 与 ground_truth patch 的行级 diff overlap
"""

import difflib
import re


class TrajectoryRewardModel:
    """规则版多维度轨迹奖励模型。

    用法:
        trm = TrajectoryRewardModel()
        scores = trm.score(trajectory)   # -> dict, 含各维度分与 total
    """

    # 各维度权重, 和为 1
    WEIGHTS = {
        "pass_rate": 0.30,
        "step_efficiency": 0.20,
        "error_recovery": 0.10,
        "localization_accuracy": 0.20,
        "patch_quality": 0.20,
    }

    # error 后向前看多少步算"恢复窗口"
    RECOVERY_WINDOW = 3

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def score(self, traj):
        """对一条轨迹打分, 返回各维度分数及加权总分。"""
        gt_file, gt_symbol = self._parse_ground_truth(traj["ground_truth_patch"])

        dims = {
            "pass_rate": self.score_pass_rate(traj),
            "step_efficiency": self.score_step_efficiency(traj, gt_file),
            "error_recovery": self.score_error_recovery(traj),
            "localization_accuracy": self.score_localization(traj, gt_file, gt_symbol),
            "patch_quality": self.score_patch_quality(traj),
        }
        dims["total"] = sum(self.WEIGHTS[k] * v for k, v in dims.items()
                            if k in self.WEIGHTS)
        return dims

    # ------------------------------------------------------------------
    # 维度 1: pass_rate
    # ------------------------------------------------------------------

    def score_pass_rate(self, traj):
        """最终测试通过率: pass -> 1.0, fail -> 0.0。"""
        return 1.0 if traj["final_result"] == "pass" else 0.0

    # ------------------------------------------------------------------
    # 维度 2: step_efficiency
    # ------------------------------------------------------------------

    def score_step_efficiency(self, traj, gt_file):
        """有效步骤占比。

        每步记分:
          - result 为 error/timeout           -> 0.0 (无效尝试)
          - 成功的 edit / test                -> 1.0 (直接推进任务)
          - 成功且命中 bug 文件的 search      -> 0.5 (有用的定位动作)
          - 其余 search / think               -> 0.0 (冗余动作)
        efficiency = 平均每步得分, 冗余步骤越多分数越低。
        """
        steps = traj["steps"]
        if not steps:
            return 0.0
        total = 0.0
        for step in steps:
            if step["result"] in ("error", "timeout"):
                continue
            if step["action"] in ("edit", "test"):
                total += 1.0
            elif step["action"] == "search" and self._touches(step, gt_file):
                total += 0.5
        return total / len(steps)

    # ------------------------------------------------------------------
    # 维度 3: error_recovery
    # ------------------------------------------------------------------

    def score_error_recovery(self, traj):
        """错误恢复能力。

        没有遇到任何错误 -> 1.0 (无需恢复)。
        对每个 error/timeout 步骤, 若其后 RECOVERY_WINDOW 步内出现成功的
        edit/test (即采取了恢复行为且成功), 视为该错误已恢复。
        得分 = 恢复的错误数 / 总错误数。
        """
        steps = traj["steps"]
        error_idxs = [i for i, s in enumerate(steps)
                      if s["result"] in ("error", "timeout")]
        if not error_idxs:
            return 1.0
        recovered = 0
        for i in error_idxs:
            window = steps[i + 1: i + 1 + self.RECOVERY_WINDOW]
            if any(s["result"] == "success" and s["action"] in ("edit", "test")
                   for s in window):
                recovered += 1
        return recovered / len(error_idxs)

    # ------------------------------------------------------------------
    # 维度 4: localization_accuracy
    # ------------------------------------------------------------------

    def score_localization(self, traj, gt_file, gt_symbol):
        """bug 定位准确率: 命中 bug 文件得 0.6, 命中函数/类名再得 0.4。"""
        score = 0.0
        if gt_file and any(self._touches(s, gt_file) for s in traj["steps"]):
            score += 0.6
        if gt_symbol and any(gt_symbol in (s.get("content") or "")
                             for s in traj["steps"]):
            score += 0.4
        return score

    # ------------------------------------------------------------------
    # 维度 5: patch_quality
    # ------------------------------------------------------------------

    def score_patch_quality(self, traj):
        """最终 patch 与 ground truth 的行级 diff overlap。

        取轨迹中最后一次成功的 edit 作为最终提交 patch, 抽取两边 patch 的
        变更行 (+/- 行), 计算:
          0.5 * Jaccard(变更行集合) + 0.5 * SequenceMatcher 序列相似度
        方向错误但删除了同样的 buggy 行 -> 部分分; 改错文件 -> 接近 0。
        """
        model_patch = self._final_edit_content(traj)
        if not model_patch:
            return 0.0
        gt_lines = self._changed_lines(traj["ground_truth_patch"])
        model_lines = self._changed_lines(model_patch)
        if not gt_lines or not model_lines:
            return 0.0

        inter = len(set(gt_lines) & set(model_lines))
        union = len(set(gt_lines) | set(model_lines))
        jaccard = inter / union if union else 0.0
        seq = difflib.SequenceMatcher(None, gt_lines, model_lines).ratio()
        return 0.5 * jaccard + 0.5 * seq

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_ground_truth(patch):
        """从 unified diff 中解析 bug 所在文件和函数/类名。"""
        gt_file = None
        gt_symbol = None
        m = re.search(r"^diff --git a/(\S+)", patch, re.M)
        if m:
            gt_file = m.group(1)
        m = re.search(r"^@@.*@@\s*(?:def|class)\s+(\w+)", patch, re.M)
        if m:
            gt_symbol = m.group(1)
        return gt_file, gt_symbol

    @staticmethod
    def _touches(step, gt_file):
        """某一步是否触碰了 ground truth bug 文件 (file 字段或 content 提到)。"""
        if not gt_file:
            return False
        return step.get("file") == gt_file or gt_file in (step.get("content") or "")

    @staticmethod
    def _final_edit_content(traj):
        """最后一次成功 edit 的内容 (视为最终提交的 patch)。"""
        for step in reversed(traj["steps"]):
            if step["action"] == "edit" and step["result"] == "success":
                return step.get("content") or ""
        return ""

    @staticmethod
    def _changed_lines(patch):
        """抽取 patch 中的变更行: '+'/'-' 开头 (排除 +++/--- 文件头), 去空白。"""
        lines = []
        for ln in patch.splitlines():
            if ln.startswith("+++") or ln.startswith("---"):
                continue
            if ln.startswith("+") or ln.startswith("-"):
                body = ln[1:].strip()
                if body:
                    lines.append(ln[0] + body)
        return lines


def traditional_reward(traj):
    """传统单一 reward: 测试通过 -> 1, 否则 -> 0 (SWE-Agent 式规则)。"""
    return 1.0 if traj["final_result"] == "pass" else 0.0


if __name__ == "__main__":
    from trajectory_data import get_trajectories

    trm = TrajectoryRewardModel()
    for traj in get_trajectories():
        dims = trm.score(traj)
        print(f"{traj['task_id']:<40} [{traj['category']}]")
        for k in ("pass_rate", "step_efficiency", "error_recovery",
                  "localization_accuracy", "patch_quality", "total"):
            print(f"    {k:<22} = {dims[k]:.3f}")
        print()
