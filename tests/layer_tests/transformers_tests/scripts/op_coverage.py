# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from primitives import make_primitive, primitive_cases
from model_primitives import implementation_inventory
from component_adapters import component_inventory


class OperatorRecorder(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.operators = set()

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        self.operators.add(str(function))
        return function(*args, **(kwargs or {}))


def source_calls(root):
    calls = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases = {"torch": "torch"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "torch" or alias.name.startswith("torch."):
                        aliases[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom) and node.module and (
                    node.module == "torch" or node.module.startswith("torch.")):
                for alias in node.names:
                    aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            parts = ast.unparse(node.func).split(".")
            if parts[0] in aliases:
                calls.append({"file": str(path.relative_to(root)), "line": node.lineno,
                              "call": ".".join([aliases[parts[0]], *parts[1:]])})
    return calls


def main():
    parser = argparse.ArgumentParser(description="Audit Transformers source call sites and primitive ATen coverage")
    parser.add_argument("--output", type=Path, required=True)
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument("--check-baseline", type=Path)
    baseline.add_argument("--update-baseline", type=Path)
    args = parser.parse_args()
    root = Path(transformers.__file__).parent.resolve()
    executed = set()
    source_prefix = str(Path(transformers.__file__).parent) + "/"
    file_names = {}

    def trace(frame, event, arg):
        if event == "line":
            filename = frame.f_code.co_filename
            if filename.startswith(source_prefix):
                if filename not in file_names:
                    file_names[filename] = str(Path(filename).resolve().relative_to(root))
                executed.add((file_names[filename], frame.f_lineno))
        return trace

    results = []
    for case in primitive_cases():
        record = {"case": f"{case.family}-{case.name}"}
        recorder = OperatorRecorder()
        try:
            model, inputs = make_primitive(case)
            previous_trace = sys.gettrace()
            try:
                sys.settrace(trace)
                with torch.no_grad(), recorder:
                    model.eval()(*inputs)
            finally:
                sys.settrace(previous_trace)
            record["status"] = "passed"
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
        record["aten_ops"] = sorted(recorder.operators)
        results.append(record)
    fixtures = json.loads((Path(__file__).resolve().parents[1] / "component_fixtures.json").read_text())
    calls = source_calls(root)
    for call in calls:
        call["executed"] = (call["file"], call["line"]) in executed
    report = {"transformers_version": transformers.__version__, "torch_version": torch.__version__,
              "cases": results, "source_torch_calls": calls,
              "aten_ops": sorted({op for result in results for op in result.get("aten_ops", [])}),
              "model_implementations": implementation_inventory(),
              "components": component_inventory(),
              "component_fixtures": fixtures,
              "unexecuted_source_calls": sum(not call["executed"] for call in calls)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{len(results)} cases; {len(report['aten_ops'])} ATen overloads; "
          f"{report['unexecuted_source_calls']}/{len(calls)} source call sites unexecuted")
    if any(result["status"] == "failed" for result in results):
        return 1
    components = component_inventory()
    inventory = {
        "transformers_version": transformers.__version__,
        "requirements_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(encoding="utf-8").encode()
        ).hexdigest(),
        "source_calls_sha256": hashlib.sha256(json.dumps([
            {key: call[key] for key in ("file", "line", "call")} for call in calls
        ], sort_keys=True).encode()).hexdigest(),
        "source_call_count": len(calls),
        "model_implementations": implementation_inventory(),
        "components_sha256": hashlib.sha256(json.dumps(components, sort_keys=True).encode()).hexdigest(),
        "component_count": len(components),
        "component_fixtures": fixtures,
        "cases": [{"case": result["case"], "aten_ops": result["aten_ops"]} for result in results],
    }
    if args.update_baseline:
        args.update_baseline.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    if args.check_baseline:
        expected = json.loads(args.check_baseline.read_text(encoding="utf-8"))
        changed = [key for key in inventory if inventory[key] != expected.get(key)]
        if changed:
            print(f"Coverage inventory changed: {', '.join(changed)}. Review {args.output}, add recipes for new "
                  "primitives or document exclusions in README.md, then regenerate coverage_inventory.json.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
