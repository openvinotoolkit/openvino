# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import openvino as ov

from openvino.runtime import Model, PartialShape, opset8
from openvino.runtime.passes import Manager, ModelPass, Matcher, MatcherPass, WrapType, Serialize
from openvino.runtime.utils import replace_node, check_test_pass


# Simple: for Extensions. Without any classes and inheritance.
def pattern_replacement():
    param = WrapType("Parameter")
    relu = WrapType("Relu", param.output(0))

    def callback(m: Matcher) -> bool:
        print("SimpleReplacement - Applied!")
        root = m.get_match_root()

        # Just to check that capturing works and we can
        # link pattern nodes with matched graph nodes.
        assert (relu in m.get_pattern_value_map())

        new_relu = opset8.relu(root.input_value(0))  # ot root.input(0).get_source_output()
        replace_node(root, new_relu)
        return True

    return Matcher(relu, "SimpleReplacement"), callback


# Hard: for POT. Includes all MatcherPass capabilities!
class PatternReplacement(MatcherPass):
    def __init__(self):
        MatcherPass.__init__(self)

        relu = WrapType("Relu")

        def callback(m: Matcher) -> bool:
            print("PatternReplacement - Applied!")
            root = m.get_match_root()
            new_relu = opset8.relu(root.input(0).get_source_output())

            # Use new operation for additional matching
            # self.register_new_node(new_relu)

            # Input->Relu->Result => Input->Relu->Relu->Result
            root.input(0).replace_source_output(new_relu.output(0))
            return True

        self.register_matcher(Matcher(relu, "PatternReplacement"), callback)


class MyModelPass(ModelPass):
    def run_on_model(self, model):
        print("ModelPass - Applied!")


def get_test_function():
    param = opset8.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    relu = opset8.relu(param.output(0))
    res = opset8.result(relu.output(0), name="result")
    return Model([res], [param], "test")


def test_transformations_api():
    check_test_pass(MyModelPass())

    m = Manager()
    #m.register_pass(MatcherPass(*pattern_replacement()))
    #m.register_pass(PatternReplacement())
    p = MyModelPass()
    m.register_pass(p)
    m.run_passes(get_test_function())

    # p = MyModelPass()
    # p.run_on_model(get_test_function())


test_transformations_api()
