# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openvino import opset8
from openvino.passes import Manager, GraphRewrite, MatcherPass, WrapType, Matcher

from tests.test_transformations.utils.utils import count_ops, get_relu_model, PatternReplacement


def test_graph_rewrite():
    model = get_relu_model()

    manager = Manager()
    # check that register pass returns pass instance
    anchor = manager.register_pass(GraphRewrite())
    anchor.add_matcher(PatternReplacement())
    manager.run_passes(model)

    assert count_ops(model, "Relu") == [2]


def test_register_new_node():
    class InsertExp(MatcherPass):
        def __init__(self):
            MatcherPass.__init__(self)
            self.model_changed = False

            param = WrapType("opset8.Parameter")

            def callback(matcher: Matcher) -> bool:
                # Input->...->Result => Input->Exp->...->Result
                root = matcher.get_match_value()
                consumers = root.get_target_inputs()

                exp = opset8.exp(root)
                for consumer in consumers:
                    consumer.replace_source_output(exp.output(0))

                # For testing purpose
                self.model_changed = True

                # Use new operation for additional matching
                self.register_new_node(exp)

                # Root node wasn't replaced or changed
                return False

            self.register_matcher(Matcher(param, "InsertExp"), callback)

    class RemoveExp(MatcherPass):
        def __init__(self):
            MatcherPass.__init__(self)
            self.model_changed = False

            param = WrapType("opset8.Exp")

            def callback(matcher: Matcher) -> bool:
                root = matcher.get_match_root()
                root.output(0).replace(root.input_value(0))

                # For testing purpose
                self.model_changed = True

                return True

            self.register_matcher(Matcher(param, "RemoveExp"), callback)

    manager = Manager()
    ins = manager.register_pass(InsertExp())
    rem = manager.register_pass(RemoveExp())
    manager.run_passes(get_relu_model())

    assert ins.model_changed
    assert rem.model_changed
