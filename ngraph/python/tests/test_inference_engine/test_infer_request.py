import numpy as np
import os
import pytest

from openvino.inference_engine import IECore, Blob, TensorDesc

def image_path():
    path_to_repo = os.environ["DATA_PATH"]
    path_to_img = os.path.join(path_to_repo, 'validation_set', '224x224', 'dog.bmp')
    return path_to_img


def model_path(is_myriad=False):
    path_to_repo = os.environ["MODELS_PATH"]
    if not is_myriad:
        test_xml = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp32.xml')
        test_bin = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp32.bin')
    else:
        test_xml = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp16.xml')
        test_bin = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp16.bin')
    return (test_xml, test_bin)


def read_image():
    import cv2
    n, c, h, w = (1, 3, 32, 32)
    image = cv2.imread(path_to_img)
    if image is None:
        raise FileNotFoundError("Input image not found")

    image = cv2.resize(image, (h, w)) / 255
    image = image.transpose((2, 0, 1)).astype(np.float32)
    image = image.reshape((n, c, h, w))
    return image


is_myriad = os.environ.get("TEST_DEVICE") == "MYRIAD"
test_net_xml, test_net_bin = model_path(is_myriad)
path_to_img = image_path()

def test_get_perf_counts(device):
    ie_core = IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    ie_core.set_config({"PERF_COUNT": "YES"}, device)
    exec_net = ie_core.load_network(net, device)
    img = read_image()
    request = exec_net.create_infer_request()
    td = TensorDesc("FP32", [1, 3, 32, 32], "NCHW")
    input_blob = Blob(td, img)
    request.set_input({'data': input_blob})
    request.infer()
    pc = request.get_perf_counts()
    assert pc['29']["status"] == "EXECUTED"
    assert pc['29']["layer_type"] == "FullyConnected"
    del exec_net
    del ie_core
    del net


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Can't run test on device {os.environ.get('TEST_DEVICE', 'CPU')}, "
                           "Dynamic batch fully supported only on CPU")
def test_set_batch_size(device):
    ie_core = IECore()
    ie_core.set_config({"DYN_BATCH_ENABLED": "YES"}, device)
    net = ie_core.read_network(test_net_xml, test_net_bin)
    net.batch_size = 10
    data = np.zeros(shape=net.input_info['data'].input_data.shape)
    exec_net = ie_core.load_network(net, device)
    data[0] = read_image()[0]
    request = exec_net.create_infer_request()
    request.set_batch(1)
    td = TensorDesc("FP32", [1, 3, 32, 32], "NCHW")
    input_blob = Blob(td, data)
    request.set_input({'data': input_blob})
    request.infer()
    assert np.allclose(int(round(request.output_blobs['fc_out'].buffer[0][2])), 1), "Incorrect data for 1st batch"
    del exec_net
    del ie_core
    del net


def test_set_zero_batch_size(device):
    ie_core = IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    exec_net = ie_core.load_network(net, device)
    request = exec_net.create_infer_request()
    with pytest.raises(ValueError) as e:
        request.set_batch(0)
    assert "Batch size should be positive integer number but 0 specified" in str(e.value)
    del exec_net
    del ie_core
    del net


def test_set_negative_batch_size(device):
    ie_core = IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    exec_net = ie_core.load_network(net, device)
    request = exec_net.create_infer_request()
    with pytest.raises(ValueError) as e:
        request.set_batch(-1)
    assert "Batch size should be positive integer number but -1 specified" in str(e.value)
    del exec_net
    del ie_core
    del net
