"""PLAN §1 guardrails: the L3 layer is simulator-only, nothing outside tools/sim imports it, and the bridge imports it only
when L3=1"""
import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

L3_DIR = Path(__file__).resolve().parents[1]           # openpilot/tools/sim/l3
SIM_DIR = L3_DIR.parent                                 # openpilot/tools/sim
PKG_DIR = SIM_DIR.parents[1]                            # openpilot (the python package)
REPO = PKG_DIR.parent
BRIDGE = SIM_DIR / "bridge" / "zoomengine"
L3_IMPORT = re.compile(r"tools[./]sim[./]l3\b|from\s+openpilot\.tools\.sim\s+import\s+[^\n]*\bl3\b|import_module\([^)]*l3")


def _clean_env(**extra: str) -> dict:
  env = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "VIRTUAL_ENV") if k in os.environ}
  env["PYTHONPATH"] = str(REPO)
  env.update(extra)
  return env


def _import_l3(env: dict) -> subprocess.CompletedProcess:
  return subprocess.run([sys.executable, "-c", "import openpilot.tools.sim.l3"], cwd=REPO, env=env, capture_output=True, text=True, timeout=60)


def test_gate_refuses_without_l3():
  for env in (_clean_env(), _clean_env(L3="0"), _clean_env(L3="true")):
    r = _import_l3(env)
    assert r.returncode != 0
    assert "NotInSimulator" in r.stderr, r.stderr


def test_gate_opens_with_l3():
  r = _import_l3(_clean_env(L3="1"))
  assert r.returncode == 0, r.stderr


def test_nothing_outside_tools_sim_imports_l3():
  offenders = []
  for root, dirs, files in os.walk(PKG_DIR):
    rel = Path(root).relative_to(PKG_DIR)
    if rel.parts[:2] == ("tools", "sim"):
      dirs[:] = []
      continue
    dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "node_modules")]
    for f in files:
      if not f.endswith(".py"):
        continue
      path = Path(root) / f
      try:
        text = path.read_text(errors="ignore")
      except OSError:
        continue
      if L3_IMPORT.search(text):
        offenders.append(str(path.relative_to(PKG_DIR)))
  assert offenders == []


def _l3_gate_test(node: ast.expr) -> bool:
  """an `if` test that checks the L3 environment variable is "1": os.environ.get("L3") == "1" (or os.getenv)"""
  if not isinstance(node, ast.Compare) or len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
    return False
  sides = [node.left, node.comparators[0]]
  if not any(isinstance(s, ast.Constant) and s.value == "1" for s in sides):
    return False
  for s in sides:
    if isinstance(s, ast.Call) and s.args and isinstance(s.args[0], ast.Constant) and s.args[0].value == "L3":
      src = ast.unparse(s.func)
      if src in ("os.environ.get", "os.getenv"):
        return True
  return False


def _l3_imports(tree: ast.AST) -> list[tuple[ast.stmt, list[ast.AST]]]:
  """every import of tools.sim.l3 with its chain of ancestors"""
  found = []

  def visit(node: ast.AST, parents: list[ast.AST]) -> None:
    names = []
    if isinstance(node, ast.Import):
      names = [a.name for a in node.names]
    elif isinstance(node, ast.ImportFrom):
      mod = node.module or ""
      names = [mod] + [f"{mod}.{a.name}" for a in node.names]
    if any(re.search(r"(^|\.)tools\.sim\.l3(\.|$)", n) for n in names):
      found.append((node, parents))
    for child in ast.iter_child_nodes(node):
      visit(child, parents + [node])

  visit(tree, [])
  return found


@pytest.mark.parametrize("name", ["zoomengine_world.py", "crossroads_car.py"])
def test_bridge_imports_l3_only_behind_gate(name):
  path = BRIDGE / name
  src = path.read_text()
  tree = ast.parse(src)
  for node, parents in _l3_imports(tree):
    gated = False
    for i, p in enumerate(parents):
      child = parents[i + 1] if i + 1 < len(parents) else node
      if isinstance(p, ast.If) and _l3_gate_test(p.test) and child in p.body:
        gated = True
    assert gated, f"{name}:{node.lineno} imports tools.sim.l3 outside an L3 == '1' branch"
  # no dynamic imports that would dodge the check
  assert "importlib" not in src and "__import__" not in src


def test_bridge_modules_import_without_l3():
  """the bridge modules load, and never pull in tools.sim.l3, with L3 unset"""
  code = "\n".join(["import sys",
                    "import openpilot.tools.sim.bridge.zoomengine.zoomengine_world",
                    "import openpilot.tools.sim.bridge.zoomengine.crossroads_car",
                    "assert not [m for m in sys.modules if m.startswith('openpilot.tools.sim.l3')], 'l3 imported'"])
  r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=_clean_env(), capture_output=True, text=True, timeout=120)
  if r.returncode != 0 and "ModuleNotFoundError" in r.stderr and "l3" not in r.stderr:
    pytest.skip(f"bridge deps missing in this environment: {r.stderr.strip().splitlines()[-1]}")
  assert r.returncode == 0, r.stderr


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_no_core_diff():
  r = subprocess.run(["bash", str(L3_DIR / "tools" / "check_no_core_diff.sh")], capture_output=True, text=True, timeout=120)
  assert r.returncode == 0 and r.stdout.strip().endswith("OK"), r.stdout + r.stderr
