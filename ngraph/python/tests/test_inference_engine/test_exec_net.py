import numpy as np
import os
import pytest
import warnings

from openvino.inference_engine import IECore, IENetwork, ExecutableNetwork, DataPtr, InputInfoPtr, InputInfoCPtr


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


test_net_xml, test_net_bin = model_path()
path_to_image = image_path()


def read_image():
    import cv2
    n, c, h, w = (1, 3, 32, 32)
    image = cv2.imread(path_to_image)
    if image is None:
        raise FileNotFoundError("Input image not found")

    image = cv2.resize(image, (h, w)) / 255
    image = image.transpose((2, 0, 1))
    image = image.reshape((n, c, h, w))
    return image


# def test_infer(device):
#     ie_core = IECore()
#     net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
#     exec_net = ie_core.load_network(net, device)
#     img = read_image()
#     res = exec_net.infer({'data': img})
#     assert np.argmax(res['fc_out'][0]) == 2
#     del exec_net
#     del ie_core


def test_input_info(device):
    ie_core = IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(network=net, device_name=device)
    # print(exec_net.input_info)
    # print(exec_net.input_info['data'])
    assert (isinstance(exec_net.input_info['data'], InputInfoPtr))
    # assert isinstance(exec_net.input_info['data'], InputInfoCPtr)
    assert exec_net.input_info['data'].name == "data"
    assert exec_net.input_info['data'].precision == "FP32"
    assert isinstance(exec_net.input_info['data'].input_data, DataPtr)
    del exec_net
    del ie_core


def test_outputs(device):
    ie_core = IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(network=net, device_name=device)
    assert len(exec_net.outputs) == 1
    assert "fc_out" in exec_net.outputs
    #assert isinstance(exec_net.outputs['fc_out'], CDataPtr)


# def test_access_requests(device):
#     ie_core = IECore()
#     net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
#     exec_net = ie_core.load_network(net, device, num_requests=5)
#     assert len(exec_net.requests) == 5
#     assert isinstance(exec_net.requests[0], InferRequest)
#     del exec_net
#     del ie_core
#
#
# def test_async_infer_one_req(device):
#     ie_core = IECore()
#     net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
#     exec_net = ie_core.load_network(net, device, num_requests=1)
#     img = read_image()
#     request_handler = exec_net.start_async(request_id=0, inputs={'data': img})
#     request_handler.wait()
#     res = request_handler.output_blobs['fc_out'].buffer
#     assert np.argmax(res) == 2
#     del exec_net
#     del ie_core
#
#
# def test_async_infer_many_req(device):
#     ie_core = IECore()
#     net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
#     exec_net = ie_core.load_network(net, device, num_requests=5)
#     img = read_image()
#     for id in range(5):
#         request_handler = exec_net.start_async(request_id=id, inputs={'data': img})
#         request_handler.wait()
#         res = request_handler.output_blobs['fc_out'].buffer
#         assert np.argmax(res) == 2
#     del exec_net
#     del ie_core
#
#
# def test_async_infer_many_req_get_idle(device):
#     ie_core = IECore()
#     net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
#     num_requests = 5
#     exec_net = ie_core.load_network(net, device, num_requests=num_requests)
#     img = read_image()
#     check_id = set()
#     for id in range(2*num_requests):
#         request_id = exec_net.get_idle_request_id()
#         if request_id == -1:
#             status = exec_net.wait(num_requests=1, timeout=WaitMode.RESULT_READY)
#             assert(status == StatusCode.OK)
#         request_id = exec_net.get_idle_request_id()
#         assert(request_id >= 0)
#         request_handler = exec_net.start_async(request_id=request_id, inputs={'data': img})
#         check_id.add(request_id)
#     status = exec_net.wait(timeout=WaitMode.RESULT_READY)
#     assert status == StatusCode.OK
#     for id in range(num_requests):
#         if id in check_id:
#             assert np.argmax(exec_net.requests[id].output_blobs['fc_out'].buffer) == 2
#     del exec_net
#     del ie_core
#
#
# def test_wait_before_start(device):
#   ie_core = IECore()
#   net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
#   num_requests = 5
#   exec_net = ie_core.load_network(net, device, num_requests=num_requests)
#   img = read_image()
#   requests = exec_net.requests
#   for id in range(num_requests):
#       status = requests[id].wait()
#       assert status == StatusCode.INFER_NOT_STARTED
#       request_handler = exec_net.start_async(request_id=id, inputs={'data': img})
#       status = requests[id].wait()
#       assert status == StatusCode.OK
#       assert np.argmax(request_handler.output_blobs['fc_out'].buffer) == 2
#   del exec_net
#   del ie_core
#
#
# def test_wrong_request_id(device):
#     ie_core = IECore()
#     net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
#     exec_net = ie_core.load_network(net, device, num_requests=1)
#     img = read_image()
#     with pytest.raises(ValueError) as e:
#         exec_net.start_async(request_id=20, inputs={'data': img})
#     assert "Incorrect request_id specified!" in str(e.value)
#     del exec_net
#     del ie_core
#
#
# def test_wrong_num_requests(device):
#     with pytest.raises(ValueError) as e:
#         ie_core = IECore()
#         net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
#         ie_core.load_network(net, device, num_requests=-1)
#         assert "Incorrect number of requests specified: -1. Expected positive integer number or zero for auto detection" \
#            in str(e.value)
#         del ie_core
#
# def test_wrong_num_requests_core(device):
#     with pytest.raises(ValueError) as e:
#         ie_core = IECore()
#         net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
#         exec_net = ie_core.load_network(net, device, num_requests=-1)
#         assert "Incorrect number of requests specified: -1. Expected positive integer number or zero for auto detection" \
#            in str(e.value)
#         del ie_core
#
# def test_plugin_accessible_after_deletion(device):
#     ie_core = IECore()
#     net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
#     exec_net = ie_core.load_network(net, device)
#     img = read_image()
#     res = exec_net.infer({'data': img})
#     assert np.argmax(res['fc_out'][0]) == 2
#     del exec_net
#     del ie_core


def test_exec_graph(device):
    ie_core = IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(net, device)
    img = read_image()
    res = exec_net.infer({'data': img})
    exec_graph = exec_net.get_exec_graph_info()
    exec_graph_file = 'exec_graph.xml'
    exec_graph.serialize(exec_graph_file)
    assert os.path.exists(exec_graph_file)
    os.remove(exec_graph_file)
    del exec_net
    del exec_graph
    del ie_core


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "MYRIAD", reason="Device specific test. "
                                                                             "Only MYRIAD plugin implements network export")
def test_export_import():
    ie_core = IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(net, "MYRIAD")
    exported_net_file = 'exported_model.bin'
    exec_net.export(exported_net_file)
    assert os.path.exists(exported_net_file)
    exec_net = ie_core.import_network(exported_net_file, "MYRIAD")
    os.remove(exported_net_file)
    img = read_image()
    res = exec_net.infer({'data': img})
    assert np.argmax(res['fc_out'][0]) == 3
    del exec_net
    del ie_core


# def test_multi_out_data(device):
#     # Regression test CVS-23965
#     # Check that CDataPtr for all output layers not copied  between outputs map items
#     ie_core = IECore()
#     net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
#     net.add_outputs(['28/Reshape'])
#     exec_net = ie_core.load_network(net, device)
#     assert "fc_out" in exec_net.outputs and "28/Reshape" in exec_net.outputs
#     assert isinstance(exec_net.outputs["fc_out"], CDataPtr)
#     assert isinstance(exec_net.outputs["28/Reshape"], CDataPtr)
#     assert exec_net.outputs["fc_out"].name == "fc_out" and exec_net.outputs["fc_out"].shape == [1, 10]
#     assert exec_net.outputs["28/Reshape"].name == "28/Reshape" and exec_net.outputs["28/Reshape"].shape == [1, 5184]
#     del ie_core
#     pass
#
#
def test_get_metric(device):
    ie_core = IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(net, "CPU")
    network_name = exec_net.get_metric("NETWORK_NAME")
    assert network_name == "test_model"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU", reason="Device independent test")
def test_get_config(device):
    ie_core = IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(net, device)
    config = exec_net.get_config("PERF_COUNT")
    assert config == "NO"
