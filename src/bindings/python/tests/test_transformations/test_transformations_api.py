# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import openvino as ov

from openvino.runtime import Model, PartialShape, opset8
from openvino.runtime.passes import Manager, ModelPass, Matcher, MatcherPass, WrapType, Serialize
from openvino.runtime.utils import replace_node, check_test_pass


def get_test_function():
    param = opset8.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    relu = opset8.relu(param.output(0))
    res = opset8.result(relu.output(0), name="result")
    return Model([res], [param], "test")


def test_simple_pattern_replacement():
    # Simple: for Extensions. Without any classes and inheritance.
    def pattern_replacement():
        param = WrapType("Parameter")
        relu = WrapType("Relu", param.output(0))

        def callback(m: Matcher) -> bool:
            root = m.get_match_root()

            # Just to check that capturing works and we can
            # link pattern nodes with matched graph nodes.
            assert relu in m.get_pattern_value_map()

            new_relu = opset8.relu(root.input_value(0))  # ot root.input(0).get_source_output()
            replace_node(root, new_relu)
            return True

        return Matcher(relu, "SimpleReplacement"), callback

    m = Manager()
    m.register_pass(MatcherPass(*pattern_replacement()))
    m.run_passes(get_test_function())


def test_matcher_pass():
    # Hard: for POT. Includes all MatcherPass capabilities!
    class PatternReplacement(MatcherPass):
        def __init__(self):
            MatcherPass.__init__(self)
            self.function_changed = False

            relu = WrapType("Relu")

            def callback(m: Matcher) -> bool:
                self.applied = True
                root = m.get_match_root()
                new_relu = opset8.relu(root.input(0).get_source_output())

                # For testing purpose
                self.function_changes = True

                # Use new operation for additional matching
                # self.register_new_node(new_relu)

                # Input->Relu->Result => Input->Relu->Relu->Result
                root.input(0).replace_source_output(new_relu.output(0))
                return True

            self.register_matcher(Matcher(relu, "PatternReplacement"), callback)

    m = Manager()
    # check that register pass returns pass instance
    p = m.register_pass(PatternReplacement())
    m.run_passes(get_test_function())

    assert p.function_changed


def test_model_pass():
    class MyModelPass(ModelPass):
        def __init__(self):
            super().__init__(self)
            self.function_changed = False

        def run_on_model(self, model):
            for op in model.get_ops():
                if op.get_type_info().name == 'Relu':
                    self.function_changed = True

    m = Manager()
    p = m.register_pass(MyModelPass())
    m.run_passes(get_test_function())

    assert p.function_changed

test_simple_pattern_replacement()
test_matcher_pass()
test_model_pass()
