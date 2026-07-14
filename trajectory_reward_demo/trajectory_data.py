# -*- coding: utf-8 -*-
"""
Module 1: Trajectory 数据构造
=============================

为论文 "Self-Evolving Code Agent: Fine-grained Trajectory Reward Modeling
for Code Agent Training" 提供 demo 级别的模拟轨迹数据。

共 8 条 SWE-bench 风格的 Code Agent 执行轨迹, 覆盖 4 类对比场景:

  - efficient      x2 : 高效轨迹 (步数少、无冗余、直接定位并修复, 最终 pass)
  - redundant      x2 : 冗余轨迹 (最终 pass, 但中间有大量无效搜索和错误尝试)
  - valuable_fail  x2 : 失败但有价值 (正确定位到 bug 文件/函数, 但修复方向错误)
  - garbage        x2 : 纯垃圾轨迹 (完全没理解问题, 搜错文件、改错文件)

每条轨迹的字段:
  task_id            : SWE-bench 风格的 issue ID
  category           : 轨迹类别标签 (仅用于分析展示, 不参与打分)
  steps              : list of {action, file, content, result}
                       action ∈ {edit, search, test, think}
                       result ∈ {success, error, timeout}
  final_result       : "pass" / "fail"
  ground_truth_patch : 正确修复的 unified diff patch

约定: edit 步骤的 content 用 mini-diff 格式 (+/- 行) 表示该次编辑产生的 patch,
最后一次成功的 edit 视为该轨迹的最终提交 patch。
"""

# ---------------------------------------------------------------------------
# Ground-truth patches (unified diff 格式, hunk header 中带 def/class 便于
# reward model 解析出 bug 所在文件与函数)
# ---------------------------------------------------------------------------

GT_PATCH_SYMPY = """\
diff --git a/sympy/polys/factortools.py b/sympy/polys/factortools.py
--- a/sympy/polys/factortools.py
+++ b/sympy/polys/factortools.py
@@ -124,7 +124,7 @@ def dup_zz_mignotte_bound(f, K):
     a = dup_max_norm(f, K)
     b = abs(dup_LC(f, K))
     n = dup_degree(f)
-    return K.sqrt(K(n + 1))*2**n*a*b
+    return K(2)**n * K.sqrt(K(n + 1)) * a * b
"""

GT_PATCH_DJANGO = """\
diff --git a/django/contrib/auth/validators.py b/django/contrib/auth/validators.py
--- a/django/contrib/auth/validators.py
+++ b/django/contrib/auth/validators.py
@@ -9,7 +9,7 @@ class ASCIIUsernameValidator(validators.RegexValidator):
-    regex = r'^[\\w.@+-]+$'
+    regex = r'\\A[\\w.@+-]+\\Z'
"""

GT_PATCH_ASTROPY = """\
diff --git a/astropy/io/ascii/rst.py b/astropy/io/ascii/rst.py
--- a/astropy/io/ascii/rst.py
+++ b/astropy/io/ascii/rst.py
@@ -42,6 +42,9 @@ def write(self, lines):
-        lines = super().write(lines)
-        return [lines[1]] + lines + [lines[1]]
+        lines = super().write(lines)
+        idx = len(self.header.header_rows)
+        return [lines[idx]] + lines + [lines[idx]]
"""

GT_PATCH_REQUESTS = """\
diff --git a/requests/utils.py b/requests/utils.py
--- a/requests/utils.py
+++ b/requests/utils.py
@@ -358,10 +358,13 @@ def stream_decode_response_unicode(iterator, r):
-    if r.encoding is None:
-        for item in iterator:
-            yield item
-        return
+    if r.encoding is None:
+        encoding = r.apparent_encoding
+    else:
+        encoding = r.encoding
+    decoder = codecs.getincrementaldecoder(encoding)(errors='replace')
"""

GT_PATCH_MPL = """\
diff --git a/lib/matplotlib/__init__.py b/lib/matplotlib/__init__.py
--- a/lib/matplotlib/__init__.py
+++ b/lib/matplotlib/__init__.py
@@ -1088,7 +1088,7 @@ def get_backend():
-    return rcParams['backend']
+    return rcParams._get_backend_or_none() or rcParams['backend']
"""

GT_PATCH_SKLEARN = """\
diff --git a/sklearn/neighbors/nca.py b/sklearn/neighbors/nca.py
--- a/sklearn/neighbors/nca.py
+++ b/sklearn/neighbors/nca.py
@@ -299,7 +299,7 @@ def _validate_params(self, X, y):
-        check_scalar(self.n_components, 'n_components', int, 1)
+        check_scalar(self.n_components, 'n_components', numbers.Integral, 1)
"""

GT_PATCH_PYTEST = """\
diff --git a/src/_pytest/mark/evaluate.py b/src/_pytest/mark/evaluate.py
--- a/src/_pytest/mark/evaluate.py
+++ b/src/_pytest/mark/evaluate.py
@@ -20,13 +20,7 @@ def cached_eval(config, expr, d):
-    default = {}
-    evalcache = config._store.setdefault(evalcache_key, default)
-    try:
-        return evalcache[expr]
-    except KeyError:
-        exprcode = compile(expr, "<evalspec>", "eval")
-        evalcache[expr] = x = eval(exprcode, d)
-        return x
+    exprcode = compile(expr, "<evalspec>", "eval")
+    return eval(exprcode, d)
"""

GT_PATCH_PYLINT = """\
diff --git a/pylint/config/argument.py b/pylint/config/argument.py
--- a/pylint/config/argument.py
+++ b/pylint/config/argument.py
@@ -99,7 +99,11 @@ def _regexp_csv_transfomer(value):
-    return re.compile(value)
+    try:
+        return re.compile(value)
+    except re.error as e:
+        msg = f"Error in provided regular expression: {value}"
+        raise argparse.ArgumentTypeError(msg) from e
"""


# ---------------------------------------------------------------------------
# 8 条模拟轨迹
# ---------------------------------------------------------------------------

TRAJECTORIES = [
    # ======================================================================
    # 1-2: 高效轨迹 — 步数少、无冗余、直接定位修复, 最终 pass
    # ======================================================================
    {
        "task_id": "sympy__sympy-19254",
        "category": "efficient",
        "steps": [
            {"action": "search", "file": "sympy/polys/factortools.py",
             "content": "grep -rn 'def dup_zz_mignotte_bound' sympy/polys/",
             "result": "success"},
            {"action": "edit", "file": "sympy/polys/factortools.py",
             "content": ("@@ def dup_zz_mignotte_bound(f, K):\n"
                         "-    return K.sqrt(K(n + 1))*2**n*a*b\n"
                         "+    return K(2)**n * K.sqrt(K(n + 1)) * a * b"),
             "result": "success"},
            {"action": "test", "file": "sympy/polys/tests/test_factortools.py",
             "content": "pytest sympy/polys/tests/test_factortools.py -k mignotte",
             "result": "success"},
        ],
        "final_result": "pass",
        "ground_truth_patch": GT_PATCH_SYMPY,
    },
    {
        "task_id": "django__django-11099",
        "category": "efficient",
        "steps": [
            {"action": "search", "file": "django/contrib/auth/validators.py",
             "content": "grep -rn 'class ASCIIUsernameValidator' django/contrib/auth/",
             "result": "success"},
            {"action": "think", "file": None,
             "content": ("Issue 说 username 末尾带换行符也能通过校验。$ 在 Python "
                         "正则里会匹配行尾换行, 应该把 ^...$ 改成 \\A...\\Z, "
                         "改 ASCIIUsernameValidator 的 regex。"),
             "result": "success"},
            {"action": "edit", "file": "django/contrib/auth/validators.py",
             "content": ("@@ class ASCIIUsernameValidator(validators.RegexValidator):\n"
                         "-    regex = r'^[\\w.@+-]+$'\n"
                         "+    regex = r'\\A[\\w.@+-]+\\Z'"),
             "result": "success"},
            {"action": "test", "file": "tests/auth_tests/test_validators.py",
             "content": "pytest tests/auth_tests/test_validators.py",
             "result": "success"},
        ],
        "final_result": "pass",
        "ground_truth_patch": GT_PATCH_DJANGO,
    },

    # ======================================================================
    # 3-4: 冗余轨迹 — 最终 pass, 但大量无效搜索 / 错误尝试 / 反复回退
    # ======================================================================
    {
        "task_id": "astropy__astropy-14182",
        "category": "redundant",
        "steps": [
            {"action": "search", "file": "docs/io/ascii/index.rst",
             "content": "grep -rn 'header_rows' docs/",
             "result": "success"},
            {"action": "think", "file": None,
             "content": "也许是文档的问题? 先看看文档里怎么描述 RST writer 的。",
             "result": "success"},
            {"action": "search", "file": "astropy/io/ascii/core.py",
             "content": "grep -n 'header_rows' astropy/io/ascii/core.py",
             "result": "error"},
            {"action": "search", "file": "astropy/table/table.py",
             "content": "grep -n 'RST' astropy/table/table.py",
             "result": "error"},
            {"action": "search", "file": "astropy/io/ascii/fixedwidth.py",
             "content": "grep -n 'header_rows' astropy/io/ascii/fixedwidth.py",
             "result": "success"},
            {"action": "edit", "file": "astropy/io/ascii/fixedwidth.py",
             "content": ("@@ class FixedWidthHeader(basic.BasicHeader):\n"
                         "-    start_line = None\n"
                         "+    start_line = 0"),
             "result": "success"},
            {"action": "test", "file": "astropy/io/ascii/tests/test_rst.py",
             "content": "pytest astropy/io/ascii/tests/test_rst.py",
             "result": "error"},
            {"action": "edit", "file": "astropy/io/ascii/fixedwidth.py",
             "content": ("@@ class FixedWidthHeader(basic.BasicHeader):\n"
                         "-    start_line = 0\n"
                         "+    start_line = None"),
             "result": "success"},
            {"action": "think", "file": None,
             "content": "改 fixedwidth 不对, 回退了。报错栈指向 rst.py 的 write, 去看 rst.py。",
             "result": "success"},
            {"action": "search", "file": "astropy/io/ascii/rst.py",
             "content": "cat astropy/io/ascii/rst.py",
             "result": "success"},
            {"action": "edit", "file": "astropy/io/ascii/rst.py",
             "content": ("@@ def write(self, lines):\n"
                         "-        lines = super().write(lines)\n"
                         "-        return [lines[1]] + lines + [lines[1]]\n"
                         "+        lines = super().write(lines)\n"
                         "+        idx = len(self.header.header_rows)\n"
                         "+        return [lines[idx]] + lines + [lines[idx]]"),
             "result": "success"},
            {"action": "test", "file": "astropy/io/ascii/tests/test_rst.py",
             "content": "pytest astropy/io/ascii/tests/test_rst.py",
             "result": "success"},
        ],
        "final_result": "pass",
        "ground_truth_patch": GT_PATCH_ASTROPY,
    },
    {
        "task_id": "requests__requests-3362",
        "category": "redundant",
        "steps": [
            {"action": "think", "file": None,
             "content": "iter_content(decode_unicode=True) 返回 bytes? 先随便看看。",
             "result": "success"},
            {"action": "search", "file": "requests/api.py",
             "content": "grep -n 'decode_unicode' requests/api.py",
             "result": "error"},
            {"action": "search", "file": "requests/sessions.py",
             "content": "grep -n 'decode_unicode' requests/sessions.py",
             "result": "error"},
            {"action": "search", "file": "requests/adapters.py",
             "content": "grep -n 'decode_unicode' requests/adapters.py",
             "result": "error"},
            {"action": "search", "file": "requests/models.py",
             "content": "grep -n 'decode_unicode' requests/models.py",
             "result": "success"},
            {"action": "edit", "file": "requests/models.py",
             "content": ("@@ def iter_content(self, chunk_size=1, decode_unicode=False):\n"
                         "-            chunks = reused_chunks if self._content_consumed else stream_chunks\n"
                         "+            chunks = stream_chunks"),
             "result": "success"},
            {"action": "test", "file": "tests/test_requests.py",
             "content": "pytest tests/test_requests.py -k decode_unicode",
             "result": "error"},
            {"action": "edit", "file": "requests/models.py",
             "content": ("@@ def iter_content(self, chunk_size=1, decode_unicode=False):\n"
                         "-            chunks = stream_chunks\n"
                         "+            chunks = reused_chunks if self._content_consumed else stream_chunks"),
             "result": "success"},
            {"action": "search", "file": "requests/utils.py",
             "content": "grep -n 'stream_decode_response_unicode' requests/utils.py",
             "result": "success"},
            {"action": "edit", "file": "requests/utils.py",
             "content": ("@@ def stream_decode_response_unicode(iterator, r):\n"
                         "-    if r.encoding is None:\n"
                         "-        for item in iterator:\n"
                         "-            yield item\n"
                         "-        return\n"
                         "+    if r.encoding is None:\n"
                         "+        encoding = r.apparent_encoding\n"
                         "+    else:\n"
                         "+        encoding = r.encoding\n"
                         "+    decoder = codecs.getincrementaldecoder(encoding)(errors='replace')"),
             "result": "success"},
            {"action": "test", "file": "tests/test_requests.py",
             "content": "pytest tests/test_requests.py -k decode_unicode",
             "result": "success"},
        ],
        "final_result": "pass",
        "ground_truth_patch": GT_PATCH_REQUESTS,
    },

    # ======================================================================
    # 5-6: 失败但有价值 — 正确定位 bug 文件/函数, 但修复方向错误
    # ======================================================================
    {
        "task_id": "matplotlib__matplotlib-23299",
        "category": "valuable_fail",
        "steps": [
            {"action": "search", "file": "lib/matplotlib/__init__.py",
             "content": "grep -n 'def get_backend' lib/matplotlib/__init__.py",
             "result": "success"},
            {"action": "think", "file": None,
             "content": ("get_backend() 会触发 rcParams['backend'] 的 auto-resolve, "
                         "副作用把 Gcf.figs 清空了。bug 就在 get_backend 这里, "
                         "但我打算直接在函数里备份/恢复 figs 来绕过。"),
             "result": "success"},
            {"action": "edit", "file": "lib/matplotlib/__init__.py",
             "content": ("@@ def get_backend():\n"
                         "-    return rcParams['backend']\n"
                         "+    figs = dict(_pylab_helpers.Gcf.figs)\n"
                         "+    backend = rcParams['backend']\n"
                         "+    _pylab_helpers.Gcf.figs.update(figs)\n"
                         "+    return backend"),
             "result": "success"},
            {"action": "test", "file": "lib/matplotlib/tests/test_rcparams.py",
             "content": "pytest lib/matplotlib/tests/test_rcparams.py -k get_backend",
             "result": "error"},
        ],
        "final_result": "fail",
        "ground_truth_patch": GT_PATCH_MPL,
    },
    {
        "task_id": "scikit-learn__scikit-learn-14092",
        "category": "valuable_fail",
        "steps": [
            {"action": "search", "file": "sklearn/neighbors/nca.py",
             "content": "grep -n 'def _validate_params' sklearn/neighbors/nca.py",
             "result": "success"},
            {"action": "think", "file": None,
             "content": ("NCA 在 GridSearch 里挂掉是因为 _validate_params 里 "
                         "check_scalar(self.n_components, 'n_components', int, 1) "
                         "对 np.int64 报错。我先把这行校验直接删掉试试。"),
             "result": "success"},
            {"action": "edit", "file": "sklearn/neighbors/nca.py",
             "content": ("@@ def _validate_params(self, X, y):\n"
                         "-        check_scalar(self.n_components, 'n_components', int, 1)\n"
                         "+        pass  # skip n_components validation"),
             "result": "success"},
            {"action": "test", "file": "sklearn/neighbors/tests/test_nca.py",
             "content": "pytest sklearn/neighbors/tests/test_nca.py",
             "result": "error"},
        ],
        "final_result": "fail",
        "ground_truth_patch": GT_PATCH_SKLEARN,
    },

    # ======================================================================
    # 7-8: 纯垃圾轨迹 — 完全没理解问题, 搜错方向、改错文件
    # ======================================================================
    {
        "task_id": "pytest-dev__pytest-7373",
        "category": "garbage",
        "steps": [
            {"action": "think", "file": None,
             "content": "skipif 结果不对? 可能是配置文件解析的问题吧。",
             "result": "success"},
            {"action": "search", "file": "setup.py",
             "content": "grep -n 'skipif' setup.py",
             "result": "error"},
            {"action": "search", "file": "src/_pytest/config/__init__.py",
             "content": "grep -n 'cache' src/_pytest/config/__init__.py",
             "result": "success"},
            {"action": "edit", "file": "src/_pytest/config/__init__.py",
             "content": ("@@ class Config:\n"
                         "-        self._store = Store()\n"
                         "+        self._store = Store()  # TODO: maybe cache bug here?"),
             "result": "success"},
            {"action": "test", "file": "testing/test_config.py",
             "content": "pytest testing/test_config.py",
             "result": "error"},
            {"action": "search", "file": "doc/en/skipping.rst",
             "content": "grep -n 'skipif' doc/en/skipping.rst",
             "result": "success"},
        ],
        "final_result": "fail",
        "ground_truth_patch": GT_PATCH_PYTEST,
    },
    {
        "task_id": "pylint-dev__pylint-7228",
        "category": "garbage",
        "steps": [
            {"action": "think", "file": None,
             "content": "\\p{Han} 报错, 应该是用户正则写错了, 不是 pylint 的问题?",
             "result": "success"},
            {"action": "think", "file": None,
             "content": "或者是 Python 版本问题? 要不改一下 README 提示用户?",
             "result": "success"},
            {"action": "search", "file": "README.rst",
             "content": "grep -n 'regex' README.rst",
             "result": "error"},
            {"action": "edit", "file": "README.rst",
             "content": ("-Pylint is a static code analyser.\n"
                         "+Pylint is a static code analyser. Note: do not use \\p{Han} in regex."),
             "result": "success"},
            {"action": "test", "file": "tests/config/test_config.py",
             "content": "pytest tests/config/test_config.py",
             "result": "timeout"},
        ],
        "final_result": "fail",
        "ground_truth_patch": GT_PATCH_PYLINT,
    },
]


def get_trajectories():
    """返回全部 8 条模拟轨迹。"""
    return TRAJECTORIES


if __name__ == "__main__":
    print(f"共 {len(TRAJECTORIES)} 条模拟轨迹:\n")
    for traj in TRAJECTORIES:
        n_steps = len(traj["steps"])
        actions = ",".join(s["action"] for s in traj["steps"])
        print(f"  [{traj['category']:<13}] {traj['task_id']:<40} "
              f"steps={n_steps:<2} final={traj['final_result']:<4} ({actions})")
