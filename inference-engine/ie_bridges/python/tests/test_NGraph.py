from openvino.inference_engine import IENetwork

try:
    import ngraph as ng
    from ngraph.impl.op import Parameter
    from ngraph.impl import Function, Shape, Type
    ngraph_available=True
except:
    ngraph_available=False

import numpy as np
import pytest

if not ngraph_available:
    pytest.skip("NGraph is not installed, skip", allow_module_level=True)

def test_CreateIENetworkFromNGraph():
    element_type = Type.f32
    param = Parameter(element_type, Shape([1, 3, 22, 22]))
    relu = ng.relu(param)
    func = Function([relu], [param], 'test')
    caps = Function.to_capsule(func)
    cnnNetwork = IENetwork(caps)
    assert cnnNetwork != None
    assert ng.function_from_cnn(cnnNetwork) != None
    assert len(cnnNetwork.layers) == 2

def test_GetIENetworkFromNGraph():
    element_type = Type.f32
    param = Parameter(element_type, Shape([1, 3, 22, 22]))
    relu = ng.relu(param)
    func = Function([relu], [param], 'test')
    caps = Function.to_capsule(func)
    cnnNetwork = IENetwork(caps)
    assert cnnNetwork != None
    assert ng.function_from_cnn(cnnNetwork) != None
    func2 = ng.function_from_cnn(cnnNetwork)
    assert func2 != None
