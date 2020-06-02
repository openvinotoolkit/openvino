from openvino.inference_engine import IENetwork
try:
    from ngraph.impl.op import Parameter, Relu
    from ngraph.impl import Function, Shape, Type
    ngraph_available=True
except:
    ngraph_available=False

import numpy as np
import pytest

if not ngraph_available:
    pytest.skip("NGraph is not installed, skip", allow_module_level=True)

@pytest.mark.skip(reason="nGraph python API has been removed in 2020.2 LTS release")
def test_CreateIENetworkFromNGraph():
    element_type = Type.f32
    param = Parameter(element_type, Shape([1, 3, 22, 22]))
    relu = Relu(param)
    func = Function([relu], [param], 'test')
    caps = Function.to_capsule(func)
    cnnNetwork = IENetwork(caps)
    assert cnnNetwork != None
    assert cnnNetwork.get_function() != None
    assert len(cnnNetwork.layers) == 2

@pytest.mark.skip(reason="nGraph python API has been removed in 2020.2 LTS release")
def test_GetIENetworkFromNGraph():
    element_type = Type.f32
    param = Parameter(element_type, Shape([1, 3, 22, 22]))
    relu = Relu(param)
    func = Function([relu], [param], 'test')
    caps = Function.to_capsule(func)
    cnnNetwork = IENetwork(caps)
    assert cnnNetwork != None
    assert cnnNetwork.get_function() != None
    caps2 = cnnNetwork.get_function()
    func2 = Function.from_capsule(caps2)
    assert func2 != None
