from openvino.inference_engine import IECore, IENetwork
from openvino.offline_transformations import ApplyMOCTransformations, ApplyLowLatencyTransformation

import ngraph as ng
from ngraph.impl.op import Parameter
from ngraph.impl import Function, Shape, Type

from conftest import model_path


test_net_xml, test_net_bin = model_path()

def get_test_cnnnetwork():
    element_type = Type.f32
    param = Parameter(element_type, Shape([1, 3, 22, 22]))
    relu = ng.relu(param)
    func = Function([relu], [param], 'test')
    caps = Function.to_capsule(func)

    cnnNetwork = IENetwork(caps)
    assert cnnNetwork != None
    return cnnNetwork


def test_moc_transformations():
    net = get_test_cnnnetwork()
    ApplyMOCTransformations(net, False)

    f = ng.function_from_cnn(net)
    assert f != None
    assert len(f.get_ops()) == 3


def test_low_latency_transformations():
    net = get_test_cnnnetwork()
    ApplyLowLatencyTransformation(net)

    f = ng.function_from_cnn(net)
    assert f != None
    assert len(f.get_ops()) == 3
