# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openvino.runtime import Model, PartialShape, opset8
from openvino.runtime.passes import Manager, ModelPass, GraphRewrite, Matcher, MatcherPass, WrapType
from openvino.runtime.utils import replace_node, replace_output_update_name


def get_test_function():
    # Parameter->Relu->Result
    param = opset8.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    relu = opset8.relu(param.output(0))
    res = opset8.result(relu.output(0), name="result")
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


def test_output_replace():
    param = opset8.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    relu = opset8.relu(param.output(0))
    res = opset8.result(relu.output(0), name="result")

    exp = opset8.exp(param.output(0))
    relu.output(0).replace(exp.output(0))

    assert res.input_value(0).get_node() == exp


def test_replace_source_output():
    param = opset8.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    relu = opset8.relu(param.output(0))
    res = opset8.result(relu.output(0), name="result")

    exp = opset8.exp(param.output(0))
    res.input(0).replace_source_output(exp.output(0))

    assert len(exp.output(0).get_target_inputs()) == 1
    assert len(relu.output(0).get_target_inputs()) == 0
    assert next(iter(exp.output(0).get_target_inputs())).get_node() == res


def test_replace_node():
    param = opset8.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    relu = opset8.relu(param.output(0))
    res = opset8.result(relu.output(0), name="result")

    exp = opset8.exp(param.output(0))
    replace_node(relu, exp)

    assert res.input_value(0).get_node() == exp


def test_replace_output_update_name():
    param = opset8.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    relu = opset8.relu(param.output(0))
    exp = opset8.exp(relu.output(0))
    res = opset8.result(exp.output(0), name="result")

    replace_output_update_name(exp.output(0), exp.input_value(0))

    assert res.input_value(0).get_node() == exp


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

            new_relu = opset8.exp(root.input_value(0))  # ot root.input(0).get_source_output()
            replace_node(root, new_relu)
            return True

        return Matcher(relu, "SimpleReplacement"), callback

    model = get_test_function()

    m = Manager()
    m.register_pass(MatcherPass(*pattern_replacement()))
    m.run_passes(model)

    assert count_ops(model, ('Relu', 'Exp')) == [0, 1]


class PatternReplacement(MatcherPass):
    def __init__(self):
        MatcherPass.__init__(self)
        self.model_changed = False

        relu = WrapType("Relu")

        def callback(m: Matcher) -> bool:
            self.applied = True
            root = m.get_match_root()
            new_relu = opset8.relu(root.input(0).get_source_output())

            # For testing purpose
            self.model_changed = True

            # # Use new operation for additional matching
            # self.register_new_node(new_relu)

            # Input->Relu->Result => Input->Relu->Relu->Result
            root.input(0).replace_source_output(new_relu.output(0))
            return True

        self.register_matcher(Matcher(relu, "PatternReplacement"), callback)


def test_matcher_pass():
    model = get_test_function()

    m = Manager()
    # check that register pass returns pass instance
    p = m.register_pass(PatternReplacement())
    m.run_passes(model)

    assert p.model_changed
    assert count_ops(model, 'Relu') == [2]


def test_matcher_pass_apply():
    model = get_test_function()

    p = PatternReplacement()
    p.apply(model.get_result().input_value(0).get_node())

    assert count_ops(model, 'Relu') == [2]


def test_graph_rewrite():
    model = get_test_function()

    m = Manager()
    # check that register pass returns pass instance
    anchor = m.register_pass(GraphRewrite())
    anchor.add_matcher(PatternReplacement())
    m.run_passes(model)

    assert count_ops(model, 'Relu') == [2]


def test_register_new_node():
    class InsertExp(MatcherPass):
        def __init__(self):
            MatcherPass.__init__(self)
            self.model_changed = False

            param = WrapType("Parameter")

            def callback(m: Matcher) -> bool:
                # Input->...->Result => Input->Exp->...->Result
                root = m.get_match_value()
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

            param = WrapType("Exp")

            def callback(m: Matcher) -> bool:
                root = m.get_match_root()
                root.output(0).replace(root.input_value(0))

                # For testing purpose
                self.model_changed = True

                return True

            self.register_matcher(Matcher(param, "RemoveExp"), callback)

    m = Manager()
    ins = m.register_pass(InsertExp())
    rem = m.register_pass(RemoveExp())
    m.run_passes(get_test_function())

    assert ins.model_changed
    assert rem.model_changed


class MyModelPass(ModelPass):
    def __init__(self):
        super().__init__()
        self.model_changed = False

    def run_on_model(self, model):
        for op in model.get_ops():
            if op.get_type_info().name == 'Relu':
                self.model_changed = True


def test_model_pass():
    m = Manager()
    p = m.register_pass(MyModelPass())
    m.run_passes(get_test_function())

    assert p.model_changed


def test_registration_and_pass_name():
    m = Manager()

    a = m.register_pass(PatternReplacement())
    a.set_name("PatterReplacement")

    b = m.register_pass(MyModelPass())
    b.set_name("ModelPass")

    c = m.register_pass(GraphRewrite())
    c.set_name("Anchor")

    d = c.add_matcher(PatternReplacement())
    d.set_name("PatterReplacement")

    PatternReplacement().set_name("PatternReplacement")
    MyModelPass().set_name("MyModelPass")
    GraphRewrite().set_name("Anchor")
