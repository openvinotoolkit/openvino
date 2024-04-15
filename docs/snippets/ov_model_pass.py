// ! [model_pass:ov_model_pass_py]
from openvino.runtime.passes import ModelPass

class MyModelPass(ModelPass):
    def __init__(self):
        super().__init__()

    def run_on_model(self, model):
        for op in model.get_ops():
            print(op.get_friendly_name())
// ! [model_pass:ov_model_pass_py]

// ! [model_pass_full_example:ov_model_pass_py]
from openvino.runtime.passes import Manager, GraphRewrite, BackwardGraphRewrite, Serialize
from openvino import Model, PartialShape
from openvino.runtime import opset13 as ops
from openvino.runtime.passes import ModelPass, Matcher, MatcherPass, WrapType


def get_relu_model():
    # Parameter->Relu->Result
    param = ops.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    relu = ops.relu(param.output(0))
    res = ops.result(relu.output(0), name="result")
    return Model([res], [param], "test")


class MyModelPass(ModelPass):
    def __init__(self):
        super().__init__()

    def run_on_model(self, model):
        for op in model.get_ops():
            print(op.get_friendly_name())


manager = Manager()
manager.register_pass(MyModelPass())
manager.run_passes(get_relu_model())
// ! [model_pass_full_example:ov_model_pass_py]
