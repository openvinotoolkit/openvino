from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from array import array
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest  # type: ignore[import-not-found]

SCRIPT = Path(__file__).resolve().parents[1] / "spore_inject.py"


IR_TEMPLATE = textwrap.dedent(
    """
    <net name="toy" version="11">
        <inputs>
            <input id="0" name="events">
                <port id="0" precision="T2">
                    <dim>1</dim>
                    <dim>4</dim>
                </port>
            </input>
            <input id="1" name="state">
                <port id="0" precision="T2">
                    <dim>1</dim>
                    <dim>4</dim>
                </port>
            </input>
            <input id="2" name="gate_src">
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>4</dim>
                </port>
            </input>
            <input id="3" name="val_src">
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>4</dim>
                </port>
            </input>
        </inputs>
        <layers>
            <layer id="0" name="events" type="Parameter" version="opset1" precision="T2">
                <output>
                    <port id="0" precision="T2">
                        <dim>1</dim>
                        <dim>4</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="state" type="Parameter" version="opset1" precision="T2">
                <output>
                    <port id="0" precision="T2">
                        <dim>1</dim>
                        <dim>4</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" name="gate_src" type="Parameter" version="opset1" precision="FP32">
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>4</dim>
                    </port>
                </output>
            </layer>
            <layer id="3" name="val_src" type="Parameter" version="opset1" precision="FP32">
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>4</dim>
                    </port>
                </output>
            </layer>
            <layer id="4" name="gate" type="Sigmoid" version="opset1" precision="FP32">
                <input>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>4</dim>
                    </port>
                </input>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>4</dim>
                    </port>
                </output>
            </layer>
            <layer id="5" name="value_linear" type="Linear" version="opset1" precision="FP32">
                <input>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>4</dim>
                    </port>
                </input>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>4</dim>
                    </port>
                </output>
            </layer>
            <layer id="6" name="ffn_multiply" type="Multiply" version="opset1" precision="FP32">
                <input>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>4</dim>
                    </port>
                    <port id="1" precision="FP32">
                        <dim>1</dim>
                        <dim>4</dim>
                    </port>
                </input>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>4</dim>
                    </port>
                </output>
            </layer>
            <layer id="7" name="result" type="Result" version="opset1" precision="FP32">
                <input>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>4</dim>
                    </port>
                </input>
            </layer>
            <layer id="8" name="matformer_shell" type="Const" version="opset1" precision="FP32">
                <data offset="0" size="32"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>2</dim>
                        <dim>4</dim>
                    </port>
                </output>
            </layer>
        </layers>
        <edges>
            <edge from-layer="2" from-port="0" to-layer="4" to-port="0"/>
            <edge from-layer="3" from-port="0" to-layer="5" to-port="0"/>
            <edge from-layer="4" from-port="0" to-layer="6" to-port="0"/>
            <edge from-layer="5" from-port="0" to-layer="6" to-port="1"/>
            <edge from-layer="6" from-port="0" to-layer="7" to-port="0"/>
        </edges>
    </net>
    """
)


def _write_model(tmp_path: Path) -> tuple[Path, Path]:
    xml_path = tmp_path / "gemma.xml"
    xml_path.write_text(IR_TEMPLATE)
    bin_path = tmp_path / "gemma.bin"
    weights = array("f", [float(i) for i in range(8)])
    bin_path.write_bytes(weights.tobytes())
    return xml_path, bin_path


def _run_tool(args: list[str]) -> None:
    subprocess.check_call([sys.executable, str(SCRIPT), *args])


def test_inject_updates_inputs_table(tmp_path):
    xml_path, _ = _write_model(tmp_path)
    out_dir = tmp_path / "infected"
    report = tmp_path / "inject_report.json"
    _run_tool(
        [
            "--report",
            str(report),
            "inject",
            "--model",
            str(xml_path),
            "--output-dir",
            str(out_dir),
            "--geglu-pattern",
            "ffn",
        ]
    )
    summary = json.loads(report.read_text())
    assert summary and summary[0]["cyberspore_layer"].endswith("CybersporeTSSN")
    output_xml = out_dir / "gemma.chimera.xml"
    tree = ET.parse(output_xml)
    inputs_elem = tree.getroot().find("inputs")
    assert inputs_elem is not None
    input_names = [elem.attrib["name"] for elem in inputs_elem.findall("input")]
    assert any("pcn_events" in name for name in input_names)
    assert any("pcn_state" in name for name in input_names)
    assert any("pcn_selective" in name for name in input_names)


def test_prune_zeroes_shell(tmp_path):
    pytest.importorskip("numpy")
    xml_path, bin_path = _write_model(tmp_path)
    report = tmp_path / "prune_report.json"
    _run_tool(
        [
            "--report",
            str(report),
            "prune",
            "--model",
            str(xml_path),
            "--weights",
            str(bin_path),
            "--pattern",
            "matformer",
            "--shell-ratio",
            "0.5",
            "--axes",
            "1",
        ]
    )
    summary = json.loads(report.read_text())
    assert summary and summary[0]["name"] == "matformer_shell"
    pruned = array("f")
    pruned.frombytes(bin_path.read_bytes())
    assert all(value == 0.0 for value in pruned)
