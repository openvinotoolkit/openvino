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
import numpy as np
import ngraph as ng 
from ngraph.impl import Function
from openvino.inference_engine import IECore, TensorDesc, Blob, IENetwork, ExecutableNetwork
import os


def model_path(is_myriad=False):
    path_to_repo = os.environ["MODELS_PATH"]
    if not is_myriad:
        test_xml = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp32.xml')
        test_bin = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp32.bin')
    else:
        test_xml = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp16.xml')
        test_bin = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp16.bin')
    return (test_xml, test_bin)

test_net_xml, test_net_bin = model_path()


def test_blobs():
    input_shape = [1, 3, 4, 4]
    input_data_float32 = (np.random.rand(*input_shape) - 0.5).astype(np.float32)

    td = TensorDesc("FP32", input_shape, "NCHW")

    input_blob_float32 = Blob(td, input_data_float32)

    assert np.all(np.equal(input_blob_float32.buffer, input_data_float32))

    input_data_int16 = (np.random.rand(*input_shape) + 0.5).astype(np.int16)

    td = TensorDesc("I16", input_shape, "NCHW")

    input_blob_i16 = Blob(td, input_data_int16)

    assert np.all(np.equal(input_blob_i16.buffer, input_data_int16))

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
    executable_network = ie_core.load_network(cnn_network, 'CPU', {})

    td = TensorDesc("FP32", input_shape, "NCHW")

    # from IPython import embed; embed()

    request = executable_network.create_infer_request()
    input_data = np.random.rand(*input_shape) - 0.5

    expected_output = np.maximum(0.0, input_data)

    input_blob = Blob(td, input_data)

    request.set_input({'parameter': input_blob})
    request.infer()

    result = request.get_blob('relu').buffer

    assert np.allclose(result, expected_output)


def test_load_network(device):
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.load_network(net, device)
    assert isinstance(exec_net, ExecutableNetwork)


def test_read_network():
    ie_core = IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    assert isinstance(net, IENetwork)


def test_get_version(device):
    ie = IECore()
    version = ie.get_versions(device)
    assert isinstance(version, dict), "Returned version must be a dictionary"
    assert device in version, "{} plugin version wasn't found in versions"
    assert hasattr(version[device], "major"), "Returned version has no field 'major'"
    assert hasattr(version[device], "minor"), "Returned version has no field 'minor'"
    assert hasattr(version[device], "description"), "Returned version has no field 'description'"
    assert hasattr(version[device], "build_number"), "Returned version has no field 'build_number'"


def test_available_devices(device):
    ie = IECore()
    devices = ie.available_devices
    assert device in devices, "Current device '{}' is not listed in available devices '{}'".format(device,
                                                                                                   ', '.join(devices))

def test_get_config():
    ie = IECore()
    conf = ie.get_config("CPU", "CPU_BIND_THREAD")
    assert conf == "YES"
