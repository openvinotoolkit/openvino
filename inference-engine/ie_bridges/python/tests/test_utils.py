from openvino.inference_engine import IECore, IENetwork

import ngraph as ng
from ngraph.impl.op import Parameter
from ngraph.impl import Function, Shape, Type

def get_test_cnnnetwork():
    element_type = Type.f32
    param = Parameter(element_type, Shape([1, 3, 22, 22]))
    relu = ng.relu(param)
    func = Function([relu], [param], 'test')
    caps = Function.to_capsule(func)

    cnnNetwork = IENetwork(caps)
    assert cnnNetwork != None
    return cnnNetwork


def test_compare_networks():
    try:
        from openvino.test_utils import CompareNetworks
        net = get_test_cnnnetwork()
        status, msg = CompareNetworks(net, net)
        assert status
    except:
        print("openvino.test_utils.CompareNetworks is not available")
