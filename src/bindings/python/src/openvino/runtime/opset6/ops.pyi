from __future__ import annotations
from openvino.opset6.ops import ctc_greedy_decoder_seq_len
from openvino.opset6.ops import gather_elements
from openvino.opset6.ops import mvn
import openvino.utils.decorators
__all__ = ['ctc_greedy_decoder_seq_len', 'gather_elements', 'mvn', 'read_value']
read_value: openvino.utils.decorators.MultiMethod  # value = <openvino.utils.decorators.MultiMethod object>
