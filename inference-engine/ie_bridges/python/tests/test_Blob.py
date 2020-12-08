import pytest

import numpy as np

from openvino.inference_engine import TensorDesc, Blob
from conftest import image_path


path_to_image = image_path()


def test_init_with_tensor_desc():
    tensor_desc = TensorDesc("FP32", [1, 3, 127, 127], "NHWC")
    blob = Blob(tensor_desc)
    assert isinstance(blob.buffer, np.ndarray)
    assert blob.tensor_desc == tensor_desc


def test_init_with_numpy():
    tensor_desc = TensorDesc("FP32", [1, 3, 127, 127], "NCHW")
    array = np.ones(shape=(1, 3, 127, 127), dtype=np.float32)
    blob = Blob(tensor_desc, array)
    assert isinstance(blob.buffer, np.ndarray)
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


def test_write_to_buffer_fp32():
    tensor_desc = TensorDesc("FP32", [1, 3, 127, 127], "NCHW")
    array = np.zeros(shape=(1, 3, 127, 127), dtype=np.float32)
    blob = Blob(tensor_desc, array)
    ones_arr = np.ones(shape=(1, 3, 127, 127), dtype=np.float32)
    blob.buffer[:] = ones_arr
    assert np.array_equal(blob.buffer, ones_arr)


def test_write_to_buffer_fp64():
    tensor_desc = TensorDesc("FP64", [1, 3, 127, 127], "NCHW")
    array = np.zeros(shape=(1, 3, 127, 127), dtype=np.float64)
    blob = Blob(tensor_desc, array)
    ones_arr = np.ones(shape=(1, 3, 127, 127), dtype=np.float64)
    blob.buffer[:] = ones_arr
    assert np.array_equal(blob.buffer, ones_arr)


@pytest.mark.skip(reason="Need to figure out how to implement right conversion")
def test_write_to_buffer_fp16():
    tensor_desc = TensorDesc("FP16", [1, 3, 127, 127], "NCHW")
    array = np.zeros(shape=(1, 3, 127, 127), dtype=np.float16)
    blob = Blob(tensor_desc, array)
    ones_arr = np.ones(shape=(1, 3, 127, 127), dtype=np.float16)
    blob.buffer[:] = ones_arr
    assert np.array_equal(blob.buffer, ones_arr)


def test_write_to_buffer_int8():
    tensor_desc = TensorDesc("I8", [1, 3, 127, 127], "NCHW")
    array = np.zeros(shape=(1, 3, 127, 127), dtype=np.int8)
    blob = Blob(tensor_desc, array)
    ones_arr = np.ones(shape=(1, 3, 127, 127), dtype=np.int8)
    blob.buffer[:] = ones_arr
    assert np.array_equal(blob.buffer, ones_arr)


def test_write_to_buffer_uint8():
    tensor_desc = TensorDesc("U8", [1, 3, 127, 127], "NCHW")
    array = np.zeros(shape=(1, 3, 127, 127), dtype=np.uint8)
    blob = Blob(tensor_desc, array)
    ones_arr = np.ones(shape=(1, 3, 127, 127), dtype=np.uint8)
    blob.buffer[:] = ones_arr
    assert np.array_equal(blob.buffer, ones_arr)


def test_write_to_buffer_int32():
    tensor_desc = TensorDesc("I32", [1, 3, 127, 127], "NCHW")
    array = np.zeros(shape=(1, 3, 127, 127), dtype=np.int32)
    blob = Blob(tensor_desc, array)
    ones_arr = np.ones(shape=(1, 3, 127, 127), dtype=np.int32)
    blob.buffer[:] = ones_arr
    assert np.array_equal(blob.buffer, ones_arr)


def test_write_to_buffer_int16():
    tensor_desc = TensorDesc("I16", [1, 3, 127, 127], "NCHW")
    array = np.zeros(shape=(1, 3, 127, 127), dtype=np.int16)
    blob = Blob(tensor_desc, array)
    ones_arr = np.ones(shape=(1, 3, 127, 127), dtype=np.int16)
    blob.buffer[:] = ones_arr
    assert np.array_equal(blob.buffer, ones_arr)


def test_write_to_buffer_uint16():
    tensor_desc = TensorDesc("U16", [1, 3, 127, 127], "NCHW")
    array = np.zeros(shape=(1, 3, 127, 127), dtype=np.uint16)
    blob = Blob(tensor_desc, array)
    ones_arr = np.ones(shape=(1, 3, 127, 127), dtype=np.uint16)
    blob.buffer[:] = ones_arr
    assert np.array_equal(blob.buffer, ones_arr)


def test_write_to_buffer_int64():
    tensor_desc = TensorDesc("I64", [1, 3, 127, 127], "NCHW")
    array = np.zeros(shape=(1, 3, 127, 127), dtype=np.int64)
    blob = Blob(tensor_desc, array)
    ones_arr = np.ones(shape=(1, 3, 127, 127), dtype=np.int64)
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
    import cv2
    n, c, h, w = (1, 3, 32, 32)
    image = cv2.imread(path_to_image)
    if image is None:
        raise FileNotFoundError("Input image not found")

    image = cv2.resize(image, (h, w)) / 255
    image = image.transpose((2, 0, 1))
    image = image.reshape((n, c, h, w))
    tensor_desc = TensorDesc("FP32", [1, 3, 32, 32], "NCHW")
    with pytest.raises(ValueError) as e:
        Blob(tensor_desc, image)
    assert "Data type float64 of provided numpy array " \
           "doesn't match to TensorDesc precision FP32" in str(e.value)
