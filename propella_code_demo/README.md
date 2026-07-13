# PROPELLA-Code Demo

代码领域多属性质量标注(demo 级别)。核心 idea:对预训练代码数据在 5 个独立维度上
打分(而非单一总分),再用多属性组合规则做数据筛选/配比,验证其优于传统单分数
top-k 过滤。参考 PROPELLA-1(自然语言多属性标注,ICLR 2026 Workshop Oral)。

## 文件

| 文件 | 说明 |
|---|---|
| `annotation_prompt.py` | Step 1:5 维度标注 prompt(带 1/3/5 锚点的 rubric,输出严格 JSON) |
| `demo_samples.py` | Step 4:16 条 hardcode 样本(4 高质 / 5 低质 / 7 维度冲突的争议样本),运行后生成 `samples.jsonl` |
| `propella_code_demo.py` | Step 2:标注脚本,读 jsonl → 调 OpenAI 兼容 API → 输出 jsonl |
| `analyze_propella.py` | Step 3:对比 baseline(总分 top-k)与 propella(多属性规则)两种筛选策略 |

## 五个维度

| JSON key | 中文 | 度量什么 |
|---|---|---|
| `logic_density` | 逻辑密度 | 有意义的计算/控制逻辑 vs 配置/数据/样板 |
| `readability` | 可读性 | 命名、注释、结构 |
| `educational_value` | 教学性 | 是否适合作学习材料 |
| `self_containment` | 自包含性 | 脱离外部上下文能否理解 |
| `practicality` | 实用性 | 解决真实问题 vs toy example |

每维 1-5 整数分;总分 = 5 维之和(5-25),由脚本计算,不让模型做算术。

## 快速开始

```bash
cd propella_code_demo

# 1. 生成 demo 样本
python demo_samples.py samples.jsonl

# 2a. 真实 API 标注(任意 OpenAI 兼容 endpoint:OpenAI / Kimi / Doubao / vLLM ...)
export OPENAI_API_KEY=sk-xxx
python propella_code_demo.py --input samples.jsonl --output annotations.jsonl \
    --base-url https://api.openai.com/v1 --model gpt-4o-mini --workers 4

# 2b. 没有 API key 时,用 mock 模式(确定性启发式打分)跑通全流程
python propella_code_demo.py --input samples.jsonl --output annotations.jsonl --mock

# 3. 策略对比
python analyze_propella.py --input annotations.jsonl
# 自定义规则 / k:
python analyze_propella.py --input annotations.jsonl \
    --rule "logic_density>=3,self_containment>=3,practicality>=2" --topk 8
```

`--resume` 支持断点续标(跳过输出文件中已成功的 id,追加写入),上 9T 数据分片跑时有用。

## 分析输出讲了什么故事

`analyze_propella.py` 输出三部分:

1. **两个子集的维度分布对比** —— propella 选中集在逻辑密度/自包含性上系统性
   高于 baseline 选中集;
2. **维度均值差**(propella - baseline);
3. **分歧样本**(核心证据):
   - *仅 baseline 选中*:总分被可读性/教学性抬高、但逻辑密度或自包含性有硬伤的
     样本(如 `mixed_verbose_tutorial` —— 逐行注释的"加法教程",注释很好看,
     逻辑约等于零);
   - *仅 propella 选中*:可读性差拉低总分、但对预训练真正有价值的样本
     (如 `mixed_obfuscated_algo` —— 单字母变量的 Dijkstra,可读性 1 分,
     逻辑密度 5 分)。

这正是论文的 motivating example:**单一总分把互相冲突的维度平均掉了,
多属性标签保留了筛选时真正需要的信息**。

## 后续接真实数据的路径

- 把 `demo_samples.py` 换成从 9T GitHub Code 数据分片读取;
- `--base-url` 指向内部推理服务,多模型投票可在 `annotate_one` 外层套一层
  多 endpoint 取中位数;
- 用不同筛选规则(单总分 top-k vs 多属性组合 vs 维度配比)各筛一份数据,
  在 D1-D5 代码能力 BPB 体系上做下游验证。
