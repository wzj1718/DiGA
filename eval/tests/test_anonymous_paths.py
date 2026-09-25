"""Test local path configuration without importing model/GPU dependencies."""

import ast
import functools
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[1]
UTILS = (
    ROOT / "lm-evaluation-harness/lm_eval/utils.py",
    ROOT / "lmms-eval/lmms_eval/utils.py",
)


def load_path_functions(path):
    # Execute the actual loader functions in isolation from heavyweight imports.
    tree = ast.parse(path.read_text())
    names = {"env_path_constructor", "ignore_constructor", "load_yaml_config"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    nodes.extend(
        node for node in tree.body
        if isinstance(node, ast.Expr)
        and "SafeLoader.add_constructor" in ast.unparse(node)
        and "env_path_constructor" in ast.unparse(node)
    )
    namespace = {"os": os, "yaml": yaml, "functools": functools, "Path": Path}
    namespace["import_function"] = lambda loader, node, **kwargs: loader.construct_scalar(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class AnonymousPathsTest(unittest.TestCase):
    def test_defaults_overrides_includes_and_loader_modes(self):
        for path in UTILS:
            functions = load_path_functions(path)
            load = functions["load_yaml_config"]
            with tempfile.TemporaryDirectory() as temp:
                parent = Path(temp) / "parent.yaml"
                child = Path(temp) / "child.yaml"
                parent.write_text("dataset_path: !env [DIFA_DATA_ROOT, ./data, gsm8k]\n")
                child.write_text("include: parent.yaml\ntask: example\n")
                for mode in ("simple", "full"):
                    with self.subTest(harness=path.parent.name, mode=mode):
                        with patch.dict(os.environ, {}, clear=True):
                            self.assertEqual(load(yaml_path=str(child), mode=mode)["dataset_path"], "./data/gsm8k")
                        with patch.dict(os.environ, {"DIFA_DATA_ROOT": "/tmp/example data"}):
                            self.assertEqual(load(yaml_path=str(child), mode=mode)["dataset_path"], "/tmp/example data/gsm8k")
                with patch.dict(os.environ, {"DIFA_DATA_ROOT": "/tmp/example data"}):
                    self.assertEqual(yaml.safe_load(parent.read_text())["dataset_path"], "/tmp/example data/gsm8k")
                with self.assertRaises(ValueError):
                    yaml.safe_load("path: !env [DIFA_DATA_ROOT]")

    def test_all_parameterized_task_configs(self):
        for utils in UTILS:
            functions = load_path_functions(utils)
            for path in (utils.parent / "tasks").rglob("*"):
                if not path.is_file() or not (path.suffix == ".yaml" or path.name.endswith("_yaml")):
                    continue
                if "!env " not in path.read_text():
                    continue
                for mode in ("simple", "full"):
                    with self.subTest(path=path, mode=mode):
                        config = functions["load_yaml_config"](yaml_path=str(path), mode=mode)
                        self.assertIsInstance(config, dict)

    def test_notebook_outputs_are_cleared(self):
        for path in ROOT.rglob("*.ipynb"):
            notebook = json.loads(path.read_text())
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    self.assertEqual(cell.get("outputs"), [], str(path))
                    self.assertIsNone(cell.get("execution_count"), str(path))


if __name__ == "__main__":
    unittest.main()
