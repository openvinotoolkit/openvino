"""
 Copyright (C) 2018-2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from openvino.inference_engine import IECore, IENetwork
import ngraph as ng
from ngraph.impl.op import Parameter
from ngraph.impl import Function, Shape, Type

from conftest import model_path


test_net_xml, test_net_bin = model_path()


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
    assert len(ops_names) != 0
    assert 'data' in ops_names


def test_get_type_name():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    func = ng.function_from_cnn(net)
    ops = func.get_ordered_ops()
    assert ops[2].get_type_name() == "Convolution"


def test_getting_shapes():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    func = ng.function_from_cnn(net)
    ops = func.get_ordered_ops()
    shapes = [sh for sh in ops[2].shape]
    assert shapes == [1, 16, 32, 32]


def test_get_set_rt_info():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    func = ng.function_from_cnn(net)
    ops = func.get_ordered_ops()
    rt_info = ops[14].get_rt_info()
    rt_info["affinity"] = "test_affinity"
    assert ops[14].get_rt_info()["affinity"] == "test_affinity"
