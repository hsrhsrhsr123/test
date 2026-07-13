# -*- coding: utf-8 -*-
"""
PROPELLA-Code Step 4: demo 用代码样本(hardcode)。

16 条样本,三类:
  - good_*   高质量:各维度都不差,任何策略都该选中;
  - low_*    低质量:纯数据/样板/玩具,任何策略都该淘汰;
  - mixed_*  有争议:维度之间强烈冲突,是"总分筛选 vs 多属性筛选"分歧的来源,
             也是这个 demo 想讲的故事。

每条样本带一个 expected 字段(人工预判的维度画像),只用于事后对照,
不会写入 samples.jsonl,更不会喂给标注模型。

用法:
    python demo_samples.py [输出路径, 默认 samples.jsonl]
"""
import json
import sys

SAMPLES = [
    # ---------------- 高质量 ----------------
    {
        "id": "good_lru_cache",
        "expected": "各维度均高",
        "content": '''\
class LRUCache:
    """固定容量的 LRU 缓存,基于哈希表 + 双向链表,get/put 均为 O(1)。"""

    class _Node:
        __slots__ = ("key", "value", "prev", "next")

        def __init__(self, key=None, value=None):
            self.key, self.value = key, value
            self.prev = self.next = None

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._map = {}
        # 哨兵头尾节点,省去边界判断
        self._head, self._tail = self._Node(), self._Node()
        self._head.next, self._tail.prev = self._tail, self._head

    def _remove(self, node):
        node.prev.next, node.next.prev = node.next, node.prev

    def _push_front(self, node):
        node.next, node.prev = self._head.next, self._head
        self._head.next.prev = node
        self._head.next = node

    def get(self, key):
        node = self._map.get(key)
        if node is None:
            return None
        self._remove(node)
        self._push_front(node)
        return node.value

    def put(self, key, value):
        if key in self._map:
            self._remove(self._map[key])
        node = self._Node(key, value)
        self._map[key] = node
        self._push_front(node)
        if len(self._map) > self.capacity:
            lru = self._tail.prev
            self._remove(lru)
            del self._map[lru.key]
''',
    },
    {
        "id": "good_binary_search",
        "expected": "教学性/可读性尤其高",
        "content": '''\
def binary_search(sorted_list, target):
    """在升序列表中二分查找 target,返回下标,不存在返回 -1。

    经典写法要点:
    - 闭区间 [lo, hi],循环条件 lo <= hi;
    - mid 用 lo + (hi - lo) // 2 防止大数溢出(其他语言中重要)。
    """
    lo, hi = 0, len(sorted_list) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if sorted_list[mid] == target:
            return mid
        elif sorted_list[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


assert binary_search([1, 3, 5, 7, 9], 7) == 3
assert binary_search([1, 3, 5, 7, 9], 4) == -1
assert binary_search([], 1) == -1
''',
    },
    {
        "id": "good_retry_decorator",
        "expected": "实用性尤其高",
        "content": '''\
import functools
import logging
import random
import time


def retry(max_attempts=3, base_delay=1.0, max_delay=30.0, retry_on=(Exception,)):
    """指数退避 + 随机抖动的重试装饰器。

    delay = min(max_delay, base_delay * 2**attempt) * uniform(0.5, 1.5)
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except retry_on as exc:
                    if attempt == max_attempts - 1:
                        raise
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    delay *= random.uniform(0.5, 1.5)
                    logging.warning(
                        "%s failed (attempt %d/%d): %s, retrying in %.1fs",
                        fn.__name__, attempt + 1, max_attempts, exc, delay,
                    )
                    time.sleep(delay)
        return wrapper

    return decorator
''',
    },
    {
        "id": "good_topo_sort",
        "expected": "逻辑密度/教学性高",
        "content": '''\
from collections import deque


def topological_sort(num_nodes, edges):
    """Kahn 算法拓扑排序。edges 为 (u, v) 表示 u -> v。

    返回拓扑序列表;若图中存在环,抛出 ValueError。
    """
    adj = [[] for _ in range(num_nodes)]
    indegree = [0] * num_nodes
    for u, v in edges:
        adj[u].append(v)
        indegree[v] += 1

    queue = deque(i for i in range(num_nodes) if indegree[i] == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for nxt in adj[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if len(order) != num_nodes:
        raise ValueError("graph contains a cycle")
    return order
''',
    },
    # ---------------- 低质量 ----------------
    {
        "id": "low_config_dict",
        "expected": "纯配置,逻辑密度≈1",
        "content": '''\
# service configuration
CONFIG = {
    "service_name": "user-api",
    "host": "0.0.0.0",
    "port": 8080,
    "debug": False,
    "log_level": "INFO",
    "db": {
        "host": "db.internal",
        "port": 5432,
        "name": "users",
        "pool_size": 10,
        "timeout_seconds": 30,
    },
    "redis": {
        "host": "cache.internal",
        "port": 6379,
        "ttl_seconds": 600,
    },
    "feature_flags": {
        "enable_new_signup": True,
        "enable_beta_dashboard": False,
        "enable_dark_mode": True,
    },
    "cors_origins": ["https://example.com", "https://admin.example.com"],
}
''',
    },
    {
        "id": "low_autogen_pb",
        "expected": "自动生成样板,逻辑≈1、教学性≈1",
        "content": '''\
# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: user_profile.proto

_USERPROFILE = DESCRIPTOR.message_types_by_name['UserProfile']
_USERPROFILE_ADDRESS = _USERPROFILE.nested_types_by_name['Address']
UserProfile = _reflection.GeneratedProtocolMessageType('UserProfile', (_message.Message,), {
  'Address': _reflection.GeneratedProtocolMessageType('Address', (_message.Message,), {
    'DESCRIPTOR': _USERPROFILE_ADDRESS,
    '__module__': 'user_profile_pb2'
  }),
  'DESCRIPTOR': _USERPROFILE,
  '__module__': 'user_profile_pb2'
})
_sym_db.RegisterMessage(UserProfile)
_sym_db.RegisterMessage(UserProfile.Address)
_USERPROFILE._serialized_start = 47
_USERPROFILE._serialized_end = 261
_USERPROFILE_ADDRESS._serialized_start = 183
_USERPROFILE_ADDRESS._serialized_end = 261
''',
    },
    {
        "id": "low_boilerplate_getters",
        "expected": "纯样板 getter/setter",
        "content": '''\
class PersonDTO:
    def __init__(self):
        self._name = None
        self._age = None
        self._email = None
        self._phone = None

    def get_name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def get_age(self):
        return self._age

    def set_age(self, age):
        self._age = age

    def get_email(self):
        return self._email

    def set_email(self, email):
        self._email = email

    def get_phone(self):
        return self._phone

    def set_phone(self, phone):
        self._phone = phone
''',
    },
    {
        "id": "low_hello_toy",
        "expected": "玩具代码,实用性≈1",
        "content": '''\
print("hello world")
print("hello world")
print("hello world")
name = input("what is your name? ")
print("hello " + name)
print("bye")
''',
    },
    {
        "id": "low_data_table",
        "expected": "纯数据表",
        "content": '''\
COUNTRY_CODES = [
    ("AD", "Andorra", "+376"),
    ("AE", "United Arab Emirates", "+971"),
    ("AF", "Afghanistan", "+93"),
    ("AG", "Antigua and Barbuda", "+1268"),
    ("AI", "Anguilla", "+1264"),
    ("AL", "Albania", "+355"),
    ("AM", "Armenia", "+374"),
    ("AO", "Angola", "+244"),
    ("AR", "Argentina", "+54"),
    ("AT", "Austria", "+43"),
    ("AU", "Australia", "+61"),
    ("AW", "Aruba", "+297"),
    ("AZ", "Azerbaijan", "+994"),
    ("BA", "Bosnia and Herzegovina", "+387"),
    ("BB", "Barbados", "+1246"),
]
''',
    },
    # ---------------- 有争议(维度冲突) ----------------
    {
        "id": "mixed_obfuscated_algo",
        "expected": "逻辑密度 5 / 可读性 1 —— 总分中庸但对预训练有价值",
        "content": '''\
def f(a, s, t):
    import heapq
    n = len(a)
    d = [1 << 60] * n
    d[s] = 0
    q = [(0, s)]
    while q:
        c, u = heapq.heappop(q)
        if c > d[u]:
            continue
        if u == t:
            return c
        for v, w in a[u]:
            x = c + w
            if x < d[v]:
                d[v] = x
                heapq.heappush(q, (x, v))
    return -1
''',
    },
    {
        "id": "mixed_verbose_tutorial",
        "expected": "可读性/教学性高、逻辑密度 1-2 —— 总分虚高的典型",
        "content": '''\
# ============================================
# 教程:如何把两个数字加起来
# 这是一个非常详细的逐行讲解示例
# ============================================

# 第 1 步:我们先定义第一个数字,并把它存进变量 a 里
a = 3

# 第 2 步:我们再定义第二个数字,并把它存进变量 b 里
b = 5

# 第 3 步:使用加号(+)运算符,把 a 和 b 加起来
# 加号是 Python 内置的算术运算符之一
result = a + b

# 第 4 步:使用 print 函数把结果输出到屏幕上
# print 是 Python 最常用的内置函数
print("a + b =", result)

# 恭喜!你已经学会了加法运算。
# 小结:变量用来存储数据,+ 用来做加法,print 用来输出。
''',
    },
    {
        "id": "mixed_fragment_snippet",
        "expected": "逻辑不错但自包含性 1-2(大文件中截出的碎片)",
        "content": '''\
        resp = self.session.post(
            self._endpoint("/v2/orders"),
            json=serialize_order(order, schema=ORDER_SCHEMA_V2),
            timeout=self.timeout,
        )
        if resp.status_code == 409:
            existing = self._resolve_conflict(order, resp.json())
            if existing and not opts.force:
                return existing
            resp = self._retry_with_lock(order, opts)
        resp.raise_for_status()
        record = OrderRecord.from_api(resp.json(), tz=self.tz)
        self._audit.emit("order.created", record.id, actor=ctx.user_id)
        return record
''',
    },
    {
        "id": "mixed_regex_oneliner",
        "expected": "逻辑密,可读性/教学性低",
        "content": '''\
import re
slug = lambda s: re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9-]", "", re.sub(r"\\s+", "-", s.strip().lower()))).strip("-")
camel = lambda s: (lambda p: p[0] + "".join(w.title() for w in p[1:]))(re.split(r"[_\\s]+", s))
dedup = lambda xs: list(dict.fromkeys(xs))
flat = lambda xs: [y for x in xs for y in (flat(x) if isinstance(x, list) else [x])]
''',
    },
    {
        "id": "mixed_leetcode_toy",
        "expected": "可读性/教学性不错,实用性低",
        "content": '''\
def fizzbuzz(n):
    """经典 FizzBuzz:3 的倍数输出 Fizz,5 的倍数输出 Buzz,同时是则输出 FizzBuzz。"""
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result


for line in fizzbuzz(15):
    print(line)
''',
    },
    {
        "id": "mixed_sql_builder_partial",
        "expected": "实用但明显是类的一半,自包含性低",
        "content": '''\
    def build_select(self):
        parts = ["SELECT", ", ".join(self._columns or ["*"]), "FROM", self._table]
        if self._joins:
            parts.extend(self._render_join(j) for j in self._joins)
        if self._where:
            parts.append("WHERE " + " AND ".join(self._where))
        if self._group_by:
            parts.append("GROUP BY " + ", ".join(self._group_by))
        if self._order_by:
            parts.append("ORDER BY " + ", ".join(self._order_by))
        if self._limit is not None:
            parts.append(f"LIMIT {int(self._limit)}")
        return " ".join(parts), tuple(self._params)
''',
    },
    {
        "id": "mixed_numeric_no_comments",
        "expected": "算法正确但零注释、命名短",
        "content": '''\
def simp(f, a, b, n):
    if n % 2:
        n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += f(a + i * h) * (4 if i % 2 else 2)
    return s * h / 3


def newton(f, df, x0, tol=1e-10, it=100):
    x = x0
    for _ in range(it):
        fx = f(x)
        if abs(fx) < tol:
            return x
        x -= fx / df(x)
    return x
''',
    },
]


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "samples.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for s in SAMPLES:
            f.write(json.dumps({"id": s["id"], "content": s["content"]},
                               ensure_ascii=False) + "\n")
    print(f"已写入 {len(SAMPLES)} 条样本到 {out_path}")
    print("样本构成: 4 高质量(good_*) / 5 低质量(low_*) / 7 有争议(mixed_*)")


if __name__ == "__main__":
    main()
