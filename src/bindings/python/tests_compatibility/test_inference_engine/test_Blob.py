# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np
import os

from openvino.inference_engine import TensorDesc, Blob, IECore
from tests_compatibility.conftest import model_path
from ..test_utils.test_utils import generate_image  # TODO: reformat into an absolute path


def test_init_with_tensor_desc():
    tensor_desc = TensorDesc("FP32", [1, 3, 127, 127], "NHWC")
    blob = Blob(tensor_desc)
    assert isinstance(blob.buffer, np.ndarray)
    assert blob.tensor_desc == tensor_desc


@pytest.mark.parametrize("shape, layout", [
    ([1, 3, 127, 127], "NCHW"),
    ([], "SCALAR"),
])
def test_init_with_numpy(shape, layout):
    tensor_desc = TensorDesc("FP32", shape, layout)
    array = np.ones(shape=shape, dtype=np.float32)
    blob = Blob(tensor_desc, array)
    assert isinstance(blob.buffer, np.ndarray)
    assert np.shares_memory(blob.buffer, array)
    assert blob.tensor_desc == tensor_desc


def test_get_tensor_desc():
    tensor_desc = TensorDesc("FP32", [1, 127, 127, 3], "NHWC")
    blob = Blob(tensor_desc)
    assert blob.tensor_desc == tensor_desc


def test_get_buffer():
    tensor_desc = TensorDesc("FP32", [1, 3, 127, 127], "NCHW")
    array = np.ones(shape=(1, 3, 127, 127), dtype=np.float32)
    blob = Blob(tensor_desc, array)
    assert np.array_equal(blob.buffer, array)


@pytest.mark.parametrize("precision, numpy_precision", [
    ("FP32", np.float32),
    ("FP64", np.float64),
    ("FP16", np.float16),
    ("I8", np.int8),
    ("U8", np.uint8),
    ("I32", np.int32),
    ("I16", np.int16),
    ("U16", np.uint16),
    ("I64", np.int64),
    ("BOOL", np.uint8),
    ("BIN", np.int8),
    ("BF16", np.float16),
])
def test_write_to_buffer(precision, numpy_precision):
    tensor_desc = TensorDesc(precision, [1, 3, 127, 127], "NCHW")
    array = np.zeros(shape=(1, 3, 127, 127), dtype=numpy_precision)
    blob = Blob(tensor_desc, array)
    ones_arr = np.ones(shape=(1, 3, 127, 127), dtype=numpy_precision)
    blob.buffer[:] = ones_arr
    assert np.array_equal(blob.buffer, ones_arr)


def test_write_numpy_scalar_int64():
    tensor_desc = TensorDesc("I64", [], "SCALAR")
    scalar = np.array(0, dtype=np.int64)
    blob = Blob(tensor_desc, scalar)
    scalar_to_write = np.array(1, dtype=np.int64)
    blob.buffer[:] = scalar_to_write
    assert np.array_equal(blob.buffer, np.atleast_1d(scalar_to_write))


def test_incompatible_array_and_td():
    tensor_desc = TensorDesc("FP32", [1, 3, 127, 127], "NCHW")
    array = np.zeros(shape=(1, 2, 3, 4), dtype=np.float32)
    with pytest.raises(AttributeError) as e:
        Blob(tensor_desc, array)
    assert "Number of elements in provided numpy array 24 and " \
           "required by TensorDesc 48387 are not equal" in str(e.value)


def test_incompatible_input_precision():
    image = generate_image(dtype="float64") 
    tensor_desc = TensorDesc("FP32", [1, 3, 32, 32], "NCHW")
    with pytest.raises(ValueError) as e:
        Blob(tensor_desc, image)
    assert "Data type float64 of provided numpy array " \
           "doesn't match to TensorDesc precision FP32" in str(e.value)


# issue 49903
@pytest.mark.skip(reason="Test will enable when CPU fix will be merge")
@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU", reason="Device dependent test")
def test_buffer_values_after_add_outputs(device):
    test_net_xml_fp16, test_net_bin_fp16 = model_path(is_myriad=True)
    ie_core = IECore()
    if device == "CPU":
        if ie_core.get_metric(device, "FULL_DEVICE_NAME") == "arm_compute::NEON":
            pytest.skip("Can't run on ARM plugin due-to ngraph")
    net = ie_core.read_network(model=test_net_xml_fp16, weights=test_net_bin_fp16)
    output_layer = "22"
    net.add_outputs(output_layer)
    exec_net = ie_core.load_network(net, device)
    feed_dict = {
        'data': np.random.normal(0, 1, (1, 3, 32, 32)).astype(np.float32)
    }
    result = exec_net.infer(feed_dict)
    assert np.all(abs(result[output_layer])<30)
    assert result[output_layer].dtype == np.float16


def test_set_shape():
    tensor_desc = TensorDesc("FP32", [1, 3, 127, 127], "NHWC")
    blob = Blob(tensor_desc)
    blob.set_shape([1, 4, 128, 128])
    assert blob.tensor_desc.dims == [1, 4, 128, 128]
    assert blob.buffer.shape == (1, 4, 128, 128)


def test_cannot_set_shape_preallocated_memory():
    tensor_desc = TensorDesc("FP32", [1, 3, 127, 127], "NHWC")
    array = np.ones([1, 3, 127, 127], dtype=np.float32)
    blob = Blob(tensor_desc, array)
    with pytest.raises(RuntimeError) as e:
        blob.set_shape([1, 4, 128, 128])
    assert "Cannot call setShape for Blobs created on top of preallocated memory" in str(e.value)
