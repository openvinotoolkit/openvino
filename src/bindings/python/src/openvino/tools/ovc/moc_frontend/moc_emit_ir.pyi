# type: ignore
from __future__ import annotations
from openvino._ov_api import Model
from openvino.tools.ovc.moc_frontend.preprocessing import apply_preprocessing
import argparse as argparse
import openvino._ov_api
__all__ = ['Model', 'apply_preprocessing', 'argparse', 'moc_emit_ir']
def moc_emit_ir(ngraph_function: openvino._ov_api.Model, argv: argparse.Namespace):
    ...
