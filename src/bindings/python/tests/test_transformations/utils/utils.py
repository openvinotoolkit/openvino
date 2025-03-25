# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino import Model, PartialShape
from openvino import opset13 as ops
from openvino.passes import ModelPass, Matcher, MatcherPass, WrapType


def get_relu_model():
    # Parameter->Relu->Result
    param = ops.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    relu = ops.relu(param.output(0))
    res = ops.result(relu.output(0), name="result")
    return Model([res], [param], "test")


def count_ops(model, op_types):
    if isinstance(op_types, str):
        op_types = [op_types]

    cnt = [0] * len(op_types)
    types = {op_types[index]: index for index in range(len(op_types))}
    for op in model.get_ops():
        op_type = op.get_type_info().name
        if op_type in types:
            cnt[types[op_type]] += 1
    return cnt


def expect_exception(func, message=""):
    def check():
        try:
            func()
            return None
        except Exception as e:
            return str(e)
    res = check()
    if res is None:
        raise AssertionError("Exception is not thrown!")
    assert message in res


class PatternReplacement(MatcherPass):
    def __init__(self):
        MatcherPass.__init__(self)
        self.model_changed = False

        relu = WrapType("opset13::Relu")

        def callback(matcher: Matcher) -> bool:
            self.applied = True
            root = matcher.get_match_root()
            new_relu = ops.relu(root.input(0).get_source_output())

            # For testing purpose
            self.model_changed = True

            """Use new operation for additional matching
              self.register_new_node(new_relu)

              Input->Relu->Result => Input->Relu->Relu->Result
            """
            root.input(0).replace_source_output(new_relu.output(0))
            return True

        self.register_matcher(Matcher(relu, "PatternReplacement"), callback)


class MyModelPass(ModelPass):
    def __init__(self):
        super().__init__()
        self.model_changed = False

    def run_on_model(self, model):
        for op in model.get_ops():
            if op.get_type_info().name == "Relu":
                self.model_changed = True
