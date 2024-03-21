# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Package: openvino.op.util
Low level wrappers for the c++ api in ov::op::util.
"""
# flake8: noqa

from openvino._pyopenvino.op.util import UnaryElementwiseArithmetic
from openvino._pyopenvino.op.util import BinaryElementwiseComparison
from openvino._pyopenvino.op.util import BinaryElementwiseArithmetic
from openvino._pyopenvino.op.util import BinaryElementwiseLogical
from openvino._pyopenvino.op.util import ArithmeticReduction
from openvino._pyopenvino.op.util import IndexReduction
from openvino._pyopenvino.op.util import VariableInfo
from openvino._pyopenvino.op.util import Variable
from openvino._pyopenvino.op.util import MergedInputDescription
from openvino._pyopenvino.op.util import InvariantInputDescription
from openvino._pyopenvino.op.util import SliceInputDescription
from openvino._pyopenvino.op.util import ConcatOutputDescription
from openvino._pyopenvino.op.util import BodyOutputDescription
from openvino._pyopenvino.op.util import _PyOp

# TODO: Move to a dedicated file
class Op(_PyOp):
    __ops_holder = []
    def __init__(self):
        super().__init__()
        Op.__ops_holder.append(self)
    def __del__(self):
        Op.__ops_holder.remove(self)
