from openvino.inference_engine import IECore, IENetwork

try:
    import ngraph as ng
    from ngraph.impl.op import Parameter
    from ngraph.impl import Function, Shape, Type
    ngraph_available=True
except:
    ngraph_available=False

import pytest

from conftest import model_path


test_net_xml, test_net_bin = model_path()

if not ngraph_available:
    pytest.skip("NGraph is not installed, skip", allow_module_level=True)


def test_create_IENetwork_from_nGraph():
    element_type = Type.f32
    param = Parameter(element_type, Shape([1, 3, 22, 22]))
    relu = ng.relu(param)
    func = Function([relu], [param], 'test')
    caps = Function.to_capsule(func)
    cnnNetwork = IENetwork(caps)
    assert cnnNetwork != None
    func2 = ng.function_from_cnn(cnnNetwork)
    assert func2 != None
    assert len(func2.get_ops()) == 3


def test_get_IENetwork_from_nGraph():
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


def test_get_ops_from_IENetwork():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    func = ng.function_from_cnn(net)
    ops = func.get_ordered_ops()
    ops_names = [op.friendly_name for op in ops]
    assert ops_names == ['data', '20/mean/Fused_Mul_614616_const', '19/WithoutBiases', 'data_add_575/copy_const',
                         '19/Fused_Add_', '21', '22', 'onnx_initializer_node_8/Output_0/Data__const',
                         '23/WithoutBiases', '23/Dims357/copy_const', '23', '25/mean/Fused_Mul_618620_const',
                         '24/WithoutBiases', 'data_add_578583/copy_const', '24/Fused_Add_', '26', '27',
                         '28/Reshape/Cast_1955_const', '28/Reshape', 'onnx_initializer_node_17/Output_0/Data__const',
                         '29/WithoutBiases', 'onnx_initializer_node_18/Output_0/Data_/copy_const', '29', 'fc_out',
                         'fc_out/sink_port_0']
