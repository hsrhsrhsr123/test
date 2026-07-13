# -*- coding: utf-8 -*-
"""
PROPELLA-Code Step 1: 代码质量多属性标注 prompt。

设计原则(对齐 PROPELLA-1 的多属性思路):
  1. 每个维度独立打分,明确告诉模型"各维度之间不要互相妥协"——
     一段代码完全可以逻辑密度 5 分同时可读性 1 分;
  2. 每个维度给出 1/3/5 三个锚点(anchor),减少模型打分漂移;
  3. 只让模型输出各维度分数,总分由脚本计算(避免模型算术错误);
  4. 输出严格 JSON,便于批量解析。

五个维度(JSON key -> 中文名):
  logic_density      逻辑密度
  readability        可读性
  educational_value  教学性
  self_containment   自包含性
  practicality       实用性
"""

DIMENSIONS = [
    "logic_density",
    "readability",
    "educational_value",
    "self_containment",
    "practicality",
]

DIMENSION_NAMES_ZH = {
    "logic_density": "逻辑密度",
    "readability": "可读性",
    "educational_value": "教学性",
    "self_containment": "自包含性",
    "practicality": "实用性",
}

SYSTEM_PROMPT = """\
你是一名资深的代码预训练数据质量评估专家。你的任务是对给定的代码片段做**多属性**质量标注:\
在 5 个相互独立的维度上分别打 1-5 分(整数)。

重要原则:
- 各维度**互相独立**,不要因为某个维度分低就压低其他维度,也不要给出"和稀泥"的中间分。\
一段没有注释、变量全是单字母的正确算法,逻辑密度可以是 5 分,可读性同时可以是 1 分。
- 打分对象是"这段代码作为预训练语料的属性",不是"这段代码能不能上线"。

评分维度与锚点:

1. logic_density(逻辑密度):代码包含多少有意义的计算/控制逻辑,而不是配置、数据、样板代码。
   - 1 分:几乎纯数据/配置/常量表/自动生成代码,没有算法逻辑。
   - 3 分:有一定的分支、循环或数据变换,但逻辑简单直接。
   - 5 分:包含实质性算法或复杂控制流(如动态规划、图算法、状态机、精巧的位运算)。

2. readability(可读性):命名规范、注释质量、结构清晰度。
   - 1 分:变量名无意义(单字母/乱码)、无注释、结构混乱或刻意混淆。
   - 3 分:命名基本达意,结构可以跟读,注释可有可无。
   - 5 分:命名自解释、结构清晰、注释/docstring 恰到好处(注意:过度注释琐碎内容不加分)。

3. educational_value(教学性):是否适合作为学习材料,能否让读者学到通用的编程知识或模式。
   - 1 分:读完学不到任何东西(纯数据、纯样板),或写法有误导性。
   - 3 分:演示了一个常见 API 用法或普通模式,有一点参考价值。
   - 5 分:清晰地演示了一个重要算法/设计模式/最佳实践,讲解性强,适合放进教程。

4. self_containment(自包含性):不依赖外部上下文(其他文件、未定义的符号、隐含环境)就能理解。
   - 1 分:大量引用未定义的类/函数/全局状态,是明显从大文件中间截出来的碎片。
   - 3 分:有少量外部依赖(如常见第三方库),但主体逻辑在片段内闭合,能独立读懂。
   - 5 分:完全自包含,所有用到的符号都在片段内定义或来自标准库,可独立运行/理解。

5. practicality(实用性):是否在解决真实问题,而不是 toy example / 练习题。
   - 1 分:hello world、fizzbuzz 式玩具代码,或纯粹的课后练习。
   - 3 分:功能真实但场景较窄,或是常见面试题级别的实现。
   - 5 分:解决真实工程问题(重试、缓存、解析、并发控制等),可直接用于生产场景。

输出要求:
- 只输出一个 JSON 对象,不要输出任何其他文字、不要用 markdown 代码块包裹。
- JSON 必须且只能包含这些字段:logic_density, readability, educational_value, \
self_containment, practicality(均为 1-5 整数),以及 rationale(一句话中文理由,<=50 字)。
"""

USER_PROMPT_TEMPLATE = """\
请对下面的代码片段做多属性质量标注。

<code>
{code}
</code>

只输出 JSON。"""


def build_messages(code: str, max_chars: int = 8000):
    """构造 chat messages。过长代码截断(demo 级别,截断即可)。"""
    if len(code) > max_chars:
        code = code[:max_chars] + "\n... (截断)"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(code=code)},
    ]
