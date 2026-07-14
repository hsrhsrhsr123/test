# Trajectory Reward Model (TRM) Demo

论文 **"Self-Evolving Code Agent: Fine-grained Trajectory Reward Modeling for
Code Agent Training"** 的 demo 级别实现。

## 核心 idea

传统 Code Agent（如 SWE-Agent）的轨迹评估依赖固定规则（test pass/fail），
reward 只有 {0, 1} 两个取值，缺乏细粒度可学习的奖励信号。本 demo 实现
Trajectory Reward Model（TRM），对 Code Agent 执行轨迹进行多维度质量评估，
为 Agentic RL 训练提供细粒度 reward。

## 文件说明

| 文件 | 说明 |
|---|---|
| `trajectory_data.py` | Module 1: hardcode 8 条 SWE-bench 风格的模拟轨迹（高效 x2 / 冗余 x2 / 有价值失败 x2 / 垃圾 x2） |
| `trajectory_reward.py` | Module 2: 规则版多维度 TRM，5 个维度各 0-1 分 + 加权总分 |
| `analyze_trajectory_reward.py` | Module 3: 传统 pass/fail reward vs TRM reward 的对比分析与表格输出 |

## 五个 reward 维度

- **pass_rate** (0.30): 最终测试是否通过（0/1）
- **step_efficiency** (0.20): 有效步骤占比（成功的 edit/test 为有效，命中 bug 文件的 search 记半分，无效 search/think 与出错步骤为 0）
- **error_recovery** (0.10): 遇到 error/timeout 后是否在窗口内成功恢复
- **localization_accuracy** (0.20): 是否定位到 bug 所在文件（0.6）与函数/类（0.4），从 ground truth patch 自动解析
- **patch_quality** (0.20): 最终 patch 与 ground truth patch 的行级 diff overlap（Jaccard + 序列相似度）

demo 阶段全部用规则计算，不训练模型；接口与可学习 reward model 对齐
（输入一条轨迹，输出标量 reward + 维度分解）。

## 运行

```bash
python3 trajectory_data.py             # 查看 8 条轨迹概览
python3 trajectory_reward.py           # 查看每条轨迹的各维度打分
python3 analyze_trajectory_reward.py   # 完整对比分析（主入口）
```

仅依赖 Python 3 标准库。

## 关键结论（analyze 输出）

1. **高效 vs 冗余**：两者最终都 pass，传统 reward 均为 1.0 无法区分；
   TRM 通过 step_efficiency 拉开差距（0.946 vs 0.866）。
2. **有价值失败 vs 垃圾**：两者最终都 fail，传统 reward 均为 0.0 同等惩罚；
   TRM 通过 localization_accuracy 和 patch_quality 给"定位正确但修复方向
   错误"的轨迹部分分（0.339 vs 0.087），保留失败轨迹中的探索信号。
3. 传统 reward 把 8 条轨迹压缩成 2 个等价类；TRM 给出 8 个可区分的细粒度
   分数，为 RL 提供更低方差的 advantage 估计。
