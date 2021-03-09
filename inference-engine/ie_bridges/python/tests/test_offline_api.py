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
