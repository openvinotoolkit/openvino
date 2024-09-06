# ! [model_pass:ov_model_pass_py]

'''
``ModelPass`` can be used as a base class for transformation classes that take entire ``Model`` and proceed with it.
To create transformation, you need to:
1. Define a class with ``ModelPass`` as a parent.
2. Redefine the run_on_model method that will receive ``Model`` as an argument.
'''

from openvino.runtime.passes import ModelPass
from snippets import get_model

class MyModelPass(ModelPass):
    def __init__(self):
        super().__init__()

    def run_on_model(self, model):
        for op in model.get_ops():
            print(op.get_friendly_name())


'''
This example defines transformation that prints all model operation names.
The next example shows ModelPass-based transformation usage.
You create ``Model`` with ``Relu``, ``Parameter`` and ``Result`` nodes. After running this code, you will see the names of the nodes.
In order to run this script, you need to export PYTHONPATH as the path to binary OpenVINO python models.
'''

from openvino.runtime.passes import Manager, GraphRewrite, BackwardGraphRewrite, Serialize
from openvino import Model, PartialShape
from openvino.runtime import opset13 as ops
from openvino.runtime.passes import ModelPass, Matcher, MatcherPass, WrapType


class MyModelPass(ModelPass):
    def __init__(self):
        super().__init__()

    def run_on_model(self, model):
        for op in model.get_ops():
            print(op.get_friendly_name())


manager = Manager()
manager.register_pass(MyModelPass())
manager.run_passes(get_model())
# ! [model_pass:ov_model_pass_py]
