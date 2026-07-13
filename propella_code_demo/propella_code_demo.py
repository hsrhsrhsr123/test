#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROPELLA-Code Step 2: 多属性标注脚本。

从 jsonl 读取代码样本(每行 {"id": ..., "content": ...}),调用 OpenAI 兼容 API
做 5 维度质量标注,输出 jsonl(每行 id + 5 个维度分 + 总分 + rationale)。

用法:
    # 1) 生成 demo 样本
    python demo_samples.py samples.jsonl

    # 2a) 真实 API 标注(任意 OpenAI 兼容 endpoint)
    export OPENAI_API_KEY=sk-xxx
    python propella_code_demo.py --input samples.jsonl --output annotations.jsonl \\
        --base-url https://api.openai.com/v1 --model gpt-4o-mini --workers 4

    # 2b) 无 API 时用 mock 模式跑通全流程(启发式打分,仅用于演示管线)
    python propella_code_demo.py --input samples.jsonl --output annotations.jsonl --mock

仅依赖标准库。
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from annotation_prompt import DIMENSIONS, build_messages

# ---------------------------------------------------------------- API 调用


def call_chat_api(base_url, api_key, model, messages,
                  temperature=0.0, timeout=120, max_retries=3):
    """调用 OpenAI 兼容 /chat/completions,带指数退避重试,返回回复文本。"""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, KeyError, json.JSONDecodeError) as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"API 调用失败(重试 {max_retries} 次): {last_err}")


def parse_scores(text):
    """从模型回复中抽取 JSON 并校验 5 个维度分,分数裁剪到 [1, 5]。"""
    # 模型偶尔会包一层 ```json ... ``` 或夹带解释文字,取第一个 {...} 块
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"回复中找不到 JSON: {text[:200]!r}")
    obj = json.loads(match.group(0))
    scores = {}
    for dim in DIMENSIONS:
        if dim not in obj:
            raise ValueError(f"缺少维度 {dim}: {obj}")
        scores[dim] = max(1, min(5, int(round(float(obj[dim])))))
    scores["rationale"] = str(obj.get("rationale", ""))[:200]
    return scores


# ---------------------------------------------------------------- mock 标注
# 说明:mock 只是为了在没有 API key 时也能跑通"标注 -> 分析"全流程,
# 用简单启发式给出大致合理且确定性的分数,不代表真实模型判断。

_LOGIC_RE = re.compile(
    r"^\s*(if |elif |else|for |while |return|yield|raise |try:|except|with |lambda)",
    re.MULTILINE)
_KEYWORDS = {"if", "elif", "else", "for", "while", "return", "yield", "raise",
             "try", "except", "with", "lambda", "def", "class", "import",
             "from", "in", "not", "and", "or", "None", "True", "False",
             "self", "print", "assert", "del", "pass", "break", "continue"}


def _clamp(x):
    return max(1, min(5, int(round(x))))


def mock_annotate(content):
    lines = [l for l in content.splitlines() if l.strip()]
    n = max(len(lines), 1)
    comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
    has_docstring = '"""' in content or "'''" in content

    # 逻辑密度:控制流语句行占比
    logic_ratio = len(_LOGIC_RE.findall(content)) / n
    logic = _clamp(1 + logic_ratio * 9)

    # 可读性:标识符平均长度 + 注释/docstring
    idents = [w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", content)
              if w not in _KEYWORDS]
    avg_len = sum(len(w) for w in idents) / max(len(idents), 1)
    readability = _clamp(avg_len - 1 + (1 if has_docstring else 0)
                         + (1 if comment_lines / n > 0.08 else 0))

    # 教学性:可读性与逻辑的折中,docstring 加成
    edu = _clamp(0.45 * readability + 0.45 * logic + (1 if has_docstring else 0))

    # 自包含性:片段特征扣分(裸 self.、非常见本地 import、悬空缩进开头)
    self_cont = 5.0
    if "self." in content and "class " not in content:
        self_cont -= 3          # 用了 self 却没有类定义 => 明显是碎片
    if re.search(r"^(from|import)\s+(utils|helpers|common|config)\b",
                 content, re.MULTILINE):
        self_cont -= 1.5
    if content.startswith((" ", "\t")):
        self_cont -= 1          # 以缩进开头,大概率截自文件中间
    self_containment = _clamp(self_cont)

    # 实用性:玩具特征扣分,工程特征加分
    practicality = 3.0
    lowered = content.lower()
    if any(t in lowered for t in ("hello world", "fizzbuzz", "教程")):
        practicality -= 2
    if any(t in lowered for t in ("retry", "cache", "timeout", "parse",
                                  "raise", "logging", "heapq", "select")):
        practicality += 1.5
    if logic <= 1:
        practicality -= 1
    practicality = _clamp(practicality)

    return {
        "logic_density": logic,
        "readability": readability,
        "educational_value": edu,
        "self_containment": self_containment,
        "practicality": practicality,
        "rationale": "[mock] 启发式打分,仅用于演示管线",
    }


# ---------------------------------------------------------------- 主流程


def annotate_one(record, args):
    """标注单条样本,失败返回带 error 字段的结果。"""
    rid = record.get("id", "<no-id>")
    content = record.get("content", "")
    try:
        if args.mock:
            scores = mock_annotate(content)
        else:
            reply = call_chat_api(args.base_url, args.api_key, args.model,
                                  build_messages(content),
                                  timeout=args.timeout)
            scores = parse_scores(reply)
        total = sum(scores[d] for d in DIMENSIONS)
        return {"id": rid, **scores, "total": total}
    except Exception as e:  # demo:单条失败不中断整体
        print(f"[warn] 样本 {rid} 标注失败: {e}", file=sys.stderr)
        return {"id": rid, "error": str(e)}


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] 跳过第 {ln} 行(JSON 解析失败): {e}", file=sys.stderr)
    return records


def main():
    ap = argparse.ArgumentParser(description="PROPELLA-Code 多属性标注 demo")
    ap.add_argument("--input", required=True, help="输入 jsonl(id + content)")
    ap.add_argument("--output", required=True, help="输出 jsonl")
    ap.add_argument("--base-url", default=os.environ.get(
        "OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI 兼容 API base url(默认取 $OPENAI_BASE_URL)")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""),
                    help="API key(默认取 $OPENAI_API_KEY)")
    ap.add_argument("--model", default="gpt-4o-mini", help="模型名")
    ap.add_argument("--workers", type=int, default=4, help="并发数")
    ap.add_argument("--timeout", type=int, default=120, help="单次请求超时秒数")
    ap.add_argument("--mock", action="store_true",
                    help="不调 API,用启发式 mock 打分跑通流程")
    ap.add_argument("--resume", action="store_true",
                    help="断点续标:跳过输出文件里已有的 id,结果追加写入")
    args = ap.parse_args()

    if not args.mock and not args.api_key:
        ap.error("非 mock 模式需要 --api-key 或环境变量 OPENAI_API_KEY")

    records = load_jsonl(args.input)
    done_ids = set()
    if args.resume and os.path.exists(args.output):
        done_ids = {r.get("id") for r in load_jsonl(args.output)
                    if "error" not in r}
        records = [r for r in records if r.get("id") not in done_ids]
        print(f"[resume] 已有 {len(done_ids)} 条,剩余 {len(records)} 条待标注")

    print(f"共 {len(records)} 条样本,mode={'mock' if args.mock else args.model},"
          f" workers={args.workers}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda r: annotate_one(r, args), records))

    mode = "a" if (args.resume and done_ids) else "w"
    ok = 0
    with open(args.output, mode, encoding="utf-8") as f:
        for res in results:  # pool.map 保序,输出顺序与输入一致
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            ok += "error" not in res
    print(f"完成: {ok}/{len(results)} 条成功,结果已写入 {args.output}")


if __name__ == "__main__":
    main()
