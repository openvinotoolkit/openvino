# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
from openvino.inference_engine import IECore, ie_api
import numpy as np
import ngraph as ng
from openvino.inference_engine import IENetwork
from ngraph.impl import Function


def test_ie_core_class():
    input_shape = [1, 3, 4, 4]
    param = ng.parameter(input_shape, np.float32, name="parameter")
    relu = ng.relu(param, name="relu")
    func = Function([relu], [param], 'test')
    func.get_ordered_ops()[2].friendly_name = "friendly"

    capsule = Function.to_capsule(func)
    cnn_network = IENetwork(capsule)

    ie_core = IECore()
    ie_core.set_config({}, device_name='CPU')
    executable_network = ie_core.load_network(cnn_network, 'CPU')

    td = ie_api.TensorDesc("FP32", input_shape, "NCHW")

    # from IPython import embed; embed()

    request = executable_network.create_infer_request()
    input_data = np.random.rand(*input_shape) - 0.5

    expected_output = np.maximum(0.0, input_data)

    input_blob = ie_api.Blob(td, input_data) # , 0)

    request.set_input({'parameter': input_blob})
    request.infer()

    result = request.get_blob('relu').buffer()

    print(input_blob.buffer())
    print(result)

    assert np.allclose(result, expected_output)