from openvino.inference_engine import IECore, IENetwork
from openvino.offline_transformations import ApplyMOCTransformations

import ngraph as ng
from ngraph.impl.op import Parameter
from ngraph.impl import Function, Shape, Type

from conftest import model_path


test_net_xml, test_net_bin = model_path()


def test_offline_api():
    element_type = Type.f32
    param = Parameter(element_type, Shape([1, 3, 22, 22]))
    relu = ng.relu(param)
    func = Function([relu], [param], 'test')
    caps = Function.to_capsule(func)

    cnnNetwork = IENetwork(caps)
    assert cnnNetwork != None

    ApplyMOCTransformations(cnnNetwork, False)

    func2 = ng.function_from_cnn(cnnNetwork)
    assert func2 != None
    assert len(func2.get_ops()) == 3