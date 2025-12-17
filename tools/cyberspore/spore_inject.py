#!/usr/bin/env python3
"""Cyberspore infection utilities for Gemma/Gemma3 IR graphs.

Phase 3 tooling adds a PCN (Cyberspore) branch in parallel with the GeGLU FFN
blocks and optionally prunes MatFormer shells in the IR weights to prepare the
host network for incremental takeover.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from xml.etree import ElementTree as ET

LOGGER = logging.getLogger("spore-inject")

# IR helpers -----------------------------------------------------------------


def _read_dims(port: ET.Element) -> List[int]:
    dims: List[int] = []
    if port is None:
        return dims
    for dim in port.findall("dim"):
        if dim.text is None:
            continue
        dims.append(int(dim.text))
    return dims


class IRGraph:
    """Minimal XML helper for OpenVINO IR graphs."""

    def __init__(self, xml_path: Path) -> None:
        self.xml_path = xml_path
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        layers_elem = self.root.find("layers")
        edges_elem = self.root.find("edges")
        self.inputs_elem = self.root.find("inputs")
        if layers_elem is None or edges_elem is None:
            raise RuntimeError("Malformed IR: missing <layers> or <edges> sections")
        self.layers_elem = layers_elem
        self.edges_elem = edges_elem
        self.layer_map: Dict[str, ET.Element] = {
            layer.attrib["id"]: layer for layer in self.layers_elem.findall("layer")
        }
        self._next_id = max((int(idx) for idx in self.layer_map.keys()), default=0) + 1
        self._reindex_edges()

    def _reindex_edges(self) -> None:
        self.in_edges: Dict[str, List[ET.Element]] = {}
        self.out_edges: Dict[str, List[ET.Element]] = {}
        for edge in self.edges_elem.findall("edge"):
            src = edge.attrib["from-layer"]
            dst = edge.attrib["to-layer"]
            self.out_edges.setdefault(src, []).append(edge)
            self.in_edges.setdefault(dst, []).append(edge)

    def layers_by_type(self, *types: str) -> Iterable[ET.Element]:
        wanted = {t.lower() for t in types}
        for layer in self.layer_map.values():
            if layer.attrib.get("type", "").lower() in wanted:
                yield layer

    def parents(self, layer_id: str) -> List[ET.Element]:
        parents: List[ET.Element] = []
        for edge in self.in_edges.get(layer_id, []):
            parents.append(self.layer_map[edge.attrib["from-layer"]])
        return parents

    def children(self, layer_id: str) -> List[ET.Element]:
        children: List[ET.Element] = []
        for edge in self.out_edges.get(layer_id, []):
            children.append(self.layer_map[edge.attrib["to-layer"]])
        return children

    def output_dims(self, layer_id: str, port_id: str = "0") -> List[int]:
        layer = self.layer_map[layer_id]
        output = layer.find("output")
        if output is None:
            return []
        port = None
        for maybe_port in output.findall("port"):
            if maybe_port.attrib.get("id") == port_id:
                port = maybe_port
                break
        if port is None:
            return []
        return _read_dims(port)

    def allocate_id(self) -> str:
        layer_id = str(self._next_id)
        self._next_id += 1
        return layer_id

    def add_layer(self, layer: ET.Element) -> str:
        layer_id = layer.attrib.get("id")
        if not layer_id:
            layer_id = self.allocate_id()
            layer.attrib["id"] = layer_id
        self.layers_elem.append(layer)
        self.layer_map[layer_id] = layer
        return layer_id

    def add_edge(self, src: str, src_port: str, dst: str, dst_port: str) -> None:
        edge = ET.SubElement(
            self.edges_elem,
            "edge",
            {
                "from-layer": src,
                "from-port": src_port,
                "to-layer": dst,
                "to-port": dst_port,
            },
        )
        self.out_edges.setdefault(src, []).append(edge)
        self.in_edges.setdefault(dst, []).append(edge)

    def register_input(self, layer_id: str, name: str, precision: str, dims: Sequence[int]) -> None:
        if self.inputs_elem is None:
            return
        existing = [int(inp.attrib.get("id", "0")) for inp in self.inputs_elem.findall("input")]
        next_id = str(max(existing) + 1 if existing else 0)
        input_elem = ET.SubElement(self.inputs_elem, "input", {"id": next_id, "name": name})
        port = ET.SubElement(input_elem, "port", {"id": "0", "precision": precision})
        for dim in dims:
            ET.SubElement(port, "dim").text = str(dim)

    def save(self, output_xml: Path) -> None:
        self.tree.write(output_xml, encoding="UTF-8")


# GeGLU detection -------------------------------------------------------------

GATE_TYPES = {"gelu", "sigmoid", "hswish", "swish", "tanh"}
VALUE_TYPES = {"matmul", "add", "linear", "fullyconnected"}


@dataclass
class GeGLUBlock:
    multiply_id: str
    multiply_name: str
    gate_parent_id: str
    value_parent_id: str
    output_dims: List[int]


class GeGLUDetector:
    def __init__(self, graph: IRGraph, name_regex: str) -> None:
        self.graph = graph
        self.pattern = re.compile(name_regex, re.IGNORECASE)

    def find_blocks(self) -> List[GeGLUBlock]:
        blocks: List[GeGLUBlock] = []
        for layer in self.graph.layers_by_type("Multiply", "Eltwise"):
            name = layer.attrib.get("name", "")
            if not self.pattern.search(name):
                continue
            parents = self.graph.parents(layer.attrib["id"])
            if len(parents) != 2:
                continue
            types = [p.attrib.get("type", "").lower() for p in parents]
            gate_idx = next((i for i, t in enumerate(types) if t in GATE_TYPES), None)
            value_idx = next((i for i, t in enumerate(types) if t in VALUE_TYPES), None)
            if gate_idx is None or value_idx is None:
                continue
            block = GeGLUBlock(
                multiply_id=layer.attrib["id"],
                multiply_name=name,
                gate_parent_id=parents[gate_idx].attrib["id"],
                value_parent_id=parents[value_idx].attrib["id"],
                output_dims=self.graph.output_dims(layer.attrib["id"]),
            )
            blocks.append(block)
        return blocks


# Cyberspore injection --------------------------------------------------------

class CybersporeInjector:
    def __init__(self, graph: IRGraph) -> None:
        self.graph = graph

    def inject(
        self,
        blocks: Sequence[GeGLUBlock],
        homeostasis: float,
        decay: float,
        max_branches: Optional[int],
        seed_mode: str,
        dry_run: bool,
    ) -> List[Dict[str, str]]:
        applied: List[Dict[str, str]] = []
        for idx, block in enumerate(blocks):
            if max_branches is not None and idx >= max_branches:
                break
            dims = block.output_dims
            if not dims:
                LOGGER.warning("Skipping %s (no static dims)", block.multiply_name)
                continue
            if dry_run:
                applied.append(
                    {
                        "multiply": block.multiply_name,
                        "cyberspore": f"{block.multiply_name}/Cyberspore",
                        "dims": "x".join(str(d) for d in dims),
                    }
                )
                continue
            plan = self._inject_block(block, idx, dims, homeostasis, decay, seed_mode)
            applied.append(plan)
        if not dry_run:
            self.graph._reindex_edges()
        return applied

    def _inject_block(
        self,
        block: GeGLUBlock,
        ordinal: int,
        dims: List[int],
        homeostasis: float,
        decay: float,
        seed_mode: str,
    ) -> Dict[str, str]:
        suffix = f"cyber_{ordinal}"
        base_name = block.multiply_name or f"multiply_{block.multiply_id}"

        events_param = self._make_parameter(
            name=f"{base_name}/pcn_events",
            precision="T2",
            dims=dims,
            doc=f"Events seed ({seed_mode})",
        )
        state_param = self._make_parameter(
            name=f"{base_name}/pcn_state",
            precision="T2",
            dims=dims,
            doc="Recurrent state seed",
        )
        selective_param = self._make_parameter(
            name=f"{base_name}/pcn_selective",
            precision="FP32",
            dims=dims,
            doc="Selective weights",
        )

        cy_layer = ET.Element(
            "layer",
            {
                "id": self.graph.allocate_id(),
                "name": f"{base_name}/CybersporeTSSN",
                "type": "CybersporeTSSN",
                "version": "opset1",
                "precision": "T2",
            },
        )
        data = ET.SubElement(
            cy_layer,
            "data",
            {
                "homeostatic_setpoint": f"{homeostasis}",
                "decay_rate": f"{decay}",
            },
        )
        data.text = data.text  # appease linters; data already created
        in_ports = ET.SubElement(cy_layer, "input")
        for port_id in range(3):
            ET.SubElement(in_ports, "port", {"id": str(port_id)})
        out_ports = ET.SubElement(cy_layer, "output")
        out_port = ET.SubElement(out_ports, "port", {"id": "0"})
        for dim in dims:
            ET.SubElement(out_port, "dim").text = str(dim)
        cyberspore_id = self.graph.add_layer(cy_layer)

        convert_layer = ET.Element(
            "layer",
            {
                "id": self.graph.allocate_id(),
                "name": f"{base_name}/CybersporeCast",
                "type": "Convert",
                "version": "opset1",
                "precision": "FP32",
            },
        )
        ET.SubElement(convert_layer, "data", {"destination_type": "FP32"})
        conv_in = ET.SubElement(convert_layer, "input")
        ET.SubElement(conv_in, "port", {"id": "0"})
        conv_out = ET.SubElement(convert_layer, "output")
        conv_out_port = ET.SubElement(conv_out, "port", {"id": "0"})
        for dim in dims:
            ET.SubElement(conv_out_port, "dim").text = str(dim)
        convert_id = self.graph.add_layer(convert_layer)

        add_layer = ET.Element(
            "layer",
            {
                "id": self.graph.allocate_id(),
                "name": f"{base_name}/ChimeraAdd",
                "type": "Add",
                "version": "opset1",
                "precision": "FP32",
            },
        )
        ET.SubElement(add_layer, "data", {"auto_broadcast": "numpy"})
        add_input = ET.SubElement(add_layer, "input")
        ET.SubElement(add_input, "port", {"id": "0"})
        ET.SubElement(add_input, "port", {"id": "1"})
        add_output = ET.SubElement(add_layer, "output")
        add_out_port = ET.SubElement(add_output, "port", {"id": "0"})
        for dim in dims:
            ET.SubElement(add_out_port, "dim").text = str(dim)
        add_id = self.graph.add_layer(add_layer)

        # Wire parameters into Cyberspore inputs
        self.graph.add_edge(events_param, "0", cyberspore_id, "0")
        self.graph.add_edge(state_param, "0", cyberspore_id, "1")
        self.graph.add_edge(selective_param, "0", cyberspore_id, "2")
        self.graph.add_edge(cyberspore_id, "0", convert_id, "0")
        self.graph.add_edge(convert_id, "0", add_id, "1")
        self.graph.add_edge(block.multiply_id, "0", add_id, "0")

        # Redirect original consumers to Add output
        original_edges = list(self.graph.out_edges.get(block.multiply_id, []))
        for edge in original_edges:
            if edge.attrib["to-layer"] == add_id:
                continue
            edge.attrib["from-layer"] = add_id
            edge.attrib["from-port"] = "0"
            self.graph.out_edges.setdefault(add_id, []).append(edge)
        self.graph.out_edges[block.multiply_id] = [
            e for e in self.graph.out_edges.get(block.multiply_id, []) if e.attrib["to-layer"] == add_id
        ]

        return {
            "multiply": block.multiply_name,
            "cyberspore_layer": cy_layer.attrib["name"],
            "add_layer": add_layer.attrib["name"],
            "dims": "x".join(str(d) for d in dims),
        }

    def _make_parameter(self, name: str, precision: str, dims: List[int], doc: str) -> str:
        layer = ET.Element(
            "layer",
            {
                "id": self.graph.allocate_id(),
                "name": name,
                "type": "Parameter",
                "version": "opset1",
                "precision": precision,
            },
        )
        rt_info = ET.SubElement(layer, "rt_info")
        info = ET.SubElement(rt_info, "attribute", {"name": "comment", "version": "0"})
        info.text = doc
        output = ET.SubElement(layer, "output")
        port = ET.SubElement(output, "port", {"id": "0"})
        for dim in dims:
            ET.SubElement(port, "dim").text = str(dim)
        layer_id = self.graph.add_layer(layer)
        self.graph.register_input(layer_id, name, precision, dims)
        return layer_id


# MatFormer pruning -----------------------------------------------------------

class MatFormerShellPruner:
    def __init__(
        self,
        graph: IRGraph,
        weights_path: Path,
        pattern: str,
        ratio: float,
        axes: Sequence[int],
    ) -> None:
        try:
            import numpy as np  # type: ignore[import-not-found]  # pylint: disable=import-outside-toplevel
        except ImportError as exc:  # pragma: no cover - exercised via CLI error path
            raise RuntimeError("MatFormer pruning requires numpy. Please install numpy to use --prune.") from exc
        self.graph = graph
        self.weights_path = weights_path
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.ratio = ratio
        self.axes = tuple(axes)
        self._np = np
        self._dtype_map = {
            "FP32": np.float32,
            "FP16": np.float16,
            "BF16": np.uint16,
        }

    def run(self, dry_run: bool) -> List[Dict[str, str]]:
        constants = [
            layer
            for layer in self.graph.layers_by_type("Const", "Constant")
            if self.pattern.search(layer.attrib.get("name", ""))
        ]
        if not constants:
            LOGGER.warning("No MatFormer constants matched pattern '%s'", self.pattern.pattern)
            return []
        if dry_run:
            return [
                {"name": layer.attrib.get("name", ""), "offset": layer.attrib.get("data", "")}
                for layer in constants
            ]
        buffer = bytearray(self.weights_path.read_bytes())
        reports: List[Dict[str, str]] = []
        for layer in constants:
            data_attr = layer.find("data")
            if data_attr is None:
                continue
            offset = int(data_attr.attrib.get("offset", "0"))
            size = int(data_attr.attrib.get("size", "0"))
            precision = layer.attrib.get("precision", "FP32").upper()
            dtype = self._dtype_map.get(precision)
            if dtype is None:
                LOGGER.warning("Skipping %s (unsupported precision %s)", layer.attrib.get("name"), precision)
                continue
            dims = self.graph.output_dims(layer.attrib["id"])
            if not dims:
                LOGGER.warning("Skipping %s (missing dims)", layer.attrib.get("name"))
                continue
            view = memoryview(buffer)[offset : offset + size]
            arr = self._np.frombuffer(view, dtype=dtype)
            if precision == "BF16":
                arr = arr.view(self._np.uint16)
            if self._np.prod(dims) != arr.size:
                LOGGER.warning(
                    "Shape mismatch for %s: dims=%s elements=%s", layer.attrib.get("name"), dims, arr.size
                )
                continue
            reshaped = self._np.reshape(arr, dims)
            self._zero_shells(reshaped)
            reports.append(
                {
                    "name": layer.attrib.get("name", ""),
                    "offset": str(offset),
                    "size": str(size),
                }
            )
        self.weights_path.write_bytes(buffer)
        return reports

    def _zero_shells(self, tensor: Any) -> None:
        for axis in self.axes:
            dim = tensor.shape[axis]
            thickness = max(1, int(dim * self.ratio))
            slices = [slice(None)] * tensor.ndim
            slices[axis] = slice(0, thickness)
            tensor[tuple(slices)] = 0
            slices[axis] = slice(dim - thickness, dim)
            tensor[tuple(slices)] = 0


# CLI -------------------------------------------------------------------------


def _default_bin_path(xml_path: Path) -> Path:
    return xml_path.with_suffix(".bin")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gemma/Gemma3 Cyberspore infection toolkit")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--report", type=Path, help="Write JSON summary to this file")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inject = subparsers.add_parser("inject", help="Insert Cyberspore branches")
    inject.add_argument("--model", type=Path, required=True, help="Path to model.xml")
    inject.add_argument("--weights", type=Path, help="Optional model.bin (defaults to alongside XML)")
    inject.add_argument("--output-dir", type=Path, required=True, help="Directory for patched IR")
    inject.add_argument("--geglu-pattern", default="geglu|mlp|ffn", help="Regex to detect GeGLU blocks")
    inject.add_argument("--max-branches", type=int, help="Limit number of injected branches")
    inject.add_argument("--homeostasis", type=float, default=0.0, help="Cyberspore setpoint nu")
    inject.add_argument("--decay", type=float, default=0.85, help="Cyberspore decay rate")
    inject.add_argument(
        "--seed-mode",
        choices=["zeros", "ones", "gaussian"],
        default="zeros",
        help="Selective/state seed annotation",
    )
    inject.add_argument("--dry-run", action="store_true", help="Do not write files")

    prune = subparsers.add_parser("prune", help="MatFormer shell pruning")
    prune.add_argument("--model", type=Path, required=True)
    prune.add_argument("--weights", type=Path, help="Optional model.bin path")
    prune.add_argument("--pattern", default="matformer", help="Regex for Constant layer names")
    prune.add_argument("--shell-ratio", type=float, default=0.05, help="Outer shell ratio per axis")
    prune.add_argument(
        "--axes",
        type=int,
        nargs="*",
        default=[-1],
        help="Axes to prune (default: last axis)",
    )
    prune.add_argument("--dry-run", action="store_true")

    return parser


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def handle_inject(args: argparse.Namespace) -> List[Dict[str, str]]:
    graph = IRGraph(args.model)
    detector = GeGLUDetector(graph, args.geglu_pattern)
    blocks = detector.find_blocks()
    if not blocks:
        LOGGER.warning("No GeGLU blocks matched pattern '%s'", args.geglu_pattern)
        return []
    injector = CybersporeInjector(graph)
    summary = injector.inject(
        blocks,
        homeostasis=args.homeostasis,
        decay=args.decay,
        max_branches=args.max_branches,
        seed_mode=args.seed_mode,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return summary
    ensure_output_dir(args.output_dir)
    output_xml = args.output_dir / (args.model.stem + ".chimera.xml")
    graph.save(output_xml)
    weights = args.weights or _default_bin_path(args.model)
    if weights.exists():
        shutil.copy2(weights, args.output_dir / weights.name)
    LOGGER.info("Injected %d Cyberspore branches", len(summary))
    return summary


def handle_prune(args: argparse.Namespace) -> List[Dict[str, str]]:
    graph = IRGraph(args.model)
    weights = args.weights or _default_bin_path(args.model)
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights file: {weights}")
    pruner = MatFormerShellPruner(graph, weights, args.pattern, args.shell_ratio, args.axes)
    summary = pruner.run(args.dry_run)
    LOGGER.info("Pruned %d MatFormer shells", len(summary))
    return summary


COMMAND_HANDLERS = {
    "inject": handle_inject,
    "prune": handle_prune,
}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    handler = COMMAND_HANDLERS[args.command]
    summary = handler(args)
    if args.report:
        ensure_output_dir(args.report.parent)
        args.report.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
