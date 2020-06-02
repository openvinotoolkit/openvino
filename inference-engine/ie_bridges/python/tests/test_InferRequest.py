import numpy as np
import os
import pytest

from openvino.inference_engine import ie_api as ie

if os.environ.get("TEST_DEVICE") != "MYRIAD":
    test_net_xml = os.path.join(os.path.dirname(__file__), 'test_data', 'models', 'test_model.xml')
    test_net_bin = os.path.join(os.path.dirname(__file__), 'test_data', 'models', 'test_model.bin')
else:
    test_net_xml = os.path.join(os.path.dirname(__file__), 'test_data', 'models', 'test_model_fp16.xml')
    test_net_bin = os.path.join(os.path.dirname(__file__), 'test_data', 'models', 'test_model_fp16.bin')
IMAGE_PATH = os.path.join(os.path.dirname(__file__), 'test_data', 'images', 'cat3.bmp')

# computed with caffe
REF_IMAGE_RESULT = np.array([[34.6295814514, 18.9434795380, 43.2669448853, 0.4420155287, -108.4574050903,
                              -314.8240051270, 231.0738067627, -106.3504943848, 108.5880966187, 92.7254943848]])


def read_image():
    import cv2
    n, c, h, w = (1, 3, 32, 32)
    image = cv2.imread(IMAGE_PATH) / 255
    if image is None:
        raise FileNotFoundError("Input image not found")

    image = cv2.resize(image, (h, w))
    image = image.transpose((2, 0, 1))
    image = image.reshape((n, c, h, w))
    return image


def load_sample_model(device, num_requests=1):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    executable_network = ie_core.load_network(net, device, num_requests=num_requests)
    return executable_network


def test_inputs(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    executable_network = ie_core.load_network(net, device, num_requests=2)

    assert len(executable_network.requests) == 2
    for req in executable_network.requests:
        assert len(req.inputs) == 1
        assert "data" in req.inputs
        assert req.inputs['data'].shape == (1, 3, 32, 32)
    del executable_network
    del ie_core
    del net


def test_outputs(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    executable_network = ie_core.load_network(net, device, num_requests=2)

    assert len(executable_network.requests) == 2
    for req in executable_network.requests:
        assert len(req.outputs) == 1
        assert "fc_out" in req.outputs
        assert req.outputs['fc_out'].shape == (1, 10)
    del executable_network
    del ie_core
    del net


def test_inputs_list(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    executable_network = ie_core.load_network(net, device, num_requests=2)

    for req in executable_network.requests:
        assert len(req._inputs_list) == 1
        assert "data" in req._inputs_list
    del ie_core


def test_outputs_list(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    executable_network = ie_core.load_network(net, device, num_requests=2)

    for req in executable_network.requests:
        assert len(req._outputs_list) == 1
        assert "fc_out" in req._outputs_list
    del ie_core


def test_access_input_buffer(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    executable_network = ie_core.load_network(net, device, num_requests=1)
    buffer = executable_network.requests[0]._get_blob_buffer("data".encode()).to_numpy()
    assert buffer.shape == (1, 3, 32, 32)
    assert buffer.strides == (12288, 4096, 128, 4)
    assert buffer.dtype == np.float32
    del executable_network
    del ie_core
    del net


def test_access_output_buffer(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    executable_network = ie_core.load_network(net, device, num_requests=1)
    buffer = executable_network.requests[0]._get_blob_buffer("fc_out".encode()).to_numpy()
    assert buffer.shape == (1, 10)
    assert buffer.strides == (40, 4)
    assert buffer.dtype == np.float32
    del executable_network
    del ie_core
    del net


def test_write_to_inputs_directly(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    executable_network = ie_core.load_network(net, device, num_requests=1)
    img = read_image()
    executable_network.requests[0].inputs["data"][:] = img
    assert np.allclose(executable_network.requests[0].inputs["data"], img)
    del executable_network
    del ie_core
    del net


def test_write_to_inputs_copy(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    executable_network = ie_core.load_network(net, device, num_requests=1)
    img = read_image()
    request = executable_network.requests[0]
    request.inputs["data"][:] = img
    assert np.allclose(executable_network.requests[0].inputs["data"], img)
    del executable_network
    del ie_core
    del net


def test_infer(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    exec_net = ie_core.load_network(net, device, num_requests=1)
    img = read_image()
    request = exec_net.requests[0]
    request.infer({'data': img})
    res = request.outputs['fc_out']
    assert np.argmax(res[0]) == 3
    del exec_net
    del ie_core
    del net


def test_async_infer_default_timeout(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    exec_net = ie_core.load_network(net, device, num_requests=1)
    img = read_image()
    request = exec_net.requests[0]
    request.async_infer({'data': img})
    request.wait()
    res = request.outputs['fc_out']
    assert np.argmax(res[0]) == 3
    del exec_net
    del ie_core
    del net


def test_async_infer_wait_finish(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    exec_net = ie_core.load_network(net, device, num_requests=1)
    img = read_image()
    request = exec_net.requests[0]
    request.async_infer({'data': img})
    request.wait(ie.WaitMode.RESULT_READY)
    res = request.outputs['fc_out']
    assert np.argmax(res[0]) == 3
    del exec_net
    del ie_core
    del net


def test_async_infer_wait_time(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    exec_net = ie_core.load_network(net, device, num_requests=1)
    img = read_image()
    request = exec_net.requests[0]
    request.async_infer({'data': img})
    request.wait(100)
    res = request.outputs['fc_out']
    assert np.argmax(res[0]) == 3
    del exec_net
    del ie_core
    del net


def test_async_infer_wait_status(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    exec_net = ie_core.load_network(net, device, num_requests=1)
    img = read_image()
    request = exec_net.requests[0]
    request.async_infer({'data': img})
    request.wait(ie.WaitMode.RESULT_READY)
    res = request.outputs['fc_out']
    assert np.argmax(res[0]) == 3
    status = request.wait(ie.WaitMode.STATUS_ONLY)
    assert status == ie.StatusCode.OK
    del exec_net
    del ie_core
    del net


def test_async_infer_fill_inputs(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    exec_net = ie_core.load_network(net, device, num_requests=1)
    img = read_image()
    request = exec_net.requests[0]
    request.inputs['data'][:] = img
    request.async_infer()
    status_end = request.wait()
    assert status_end == ie.StatusCode.OK
    res = request.outputs['fc_out']
    assert np.argmax(res[0]) == 3
    del exec_net
    del ie_core
    del net


def test_infer_modify_outputs(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    exec_net = ie_core.load_network(net, device, num_requests=1)
    img = read_image()
    request = exec_net.requests[0]
    outputs0 = exec_net.infer({'data': img})
    status_end = request.wait()
    assert status_end == ie.StatusCode.OK
    assert np.argmax(outputs0['fc_out'][0]) == 3
    outputs0['fc_out'][0] = 0
    outputs1 = request.outputs
    assert np.argmax(outputs1['fc_out'][0]) == 3
    outputs1['fc_out'][0] = 1
    outputs2 = request.outputs
    assert np.argmax(outputs2['fc_out'][0]) == 3
    del exec_net
    del ie_core
    del net


def test_async_infer_callback(device):
    def static_vars(**kwargs):
        def decorate(func):
            for k in kwargs:
                setattr(func, k, kwargs[k])
            return func
        return decorate

    @static_vars(callback_called=0)
    def callback(self, status):
        callback.callback_called = 1

    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    exec_net = ie_core.load_network(net, device, num_requests=1)
    img = read_image()
    request = exec_net.requests[0]
    request.set_completion_callback(callback)
    request.async_infer({'data': img})
    status = request.wait()
    assert status == ie.StatusCode.OK
    res = request.outputs['fc_out']
    assert np.argmax(res[0]) == 3
    assert callback.callback_called == 1
    del exec_net
    del ie_core


def test_async_infer_callback_wait_before_start(device):
    def static_vars(**kwargs):
        def decorate(func):
            for k in kwargs:
                setattr(func, k, kwargs[k])
            return func
        return decorate

    @static_vars(callback_called=0)
    def callback(self, status):
        callback.callback_called = 1

    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    exec_net = ie_core.load_network(net, device, num_requests=1)
    img = read_image()
    request = exec_net.requests[0]
    request.set_completion_callback(callback)
    status = request.wait()
    assert status == ie.StatusCode.INFER_NOT_STARTED
    request.async_infer({'data': img})
    status = request.wait()
    assert status == ie.StatusCode.OK
    res = request.outputs['fc_out']
    assert np.argmax(res[0]) == 3
    assert callback.callback_called == 1
    del exec_net
    del ie_core


def test_async_infer_callback_wait_in_callback(device):
    class InferReqWrap:
        def __init__(self, request):
            self.request = request
            self.request.set_completion_callback(self.callback)
            self.status_code = self.request.wait(ie.WaitMode.STATUS_ONLY)
            assert self.status_code == ie.StatusCode.INFER_NOT_STARTED

        def callback(self, statusCode, userdata):
            self.status_code = self.request.wait(ie.WaitMode.STATUS_ONLY)

        def execute(self, input_data):
            self.request.async_infer(input_data)
            status = self.request.wait(ie.WaitMode.RESULT_READY)
            assert status == ie.StatusCode.OK
            assert self.status_code == ie.StatusCode.OK

    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    exec_net = ie_core.load_network(net, device, num_requests=1)
    img = read_image()
    request_wrap = InferReqWrap(exec_net.requests[0])
    request_wrap.execute({'data': img})
    del exec_net
    del ie_core

def test_get_perf_counts(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    ie_core.set_config({"PERF_COUNT": "YES"}, device)
    exec_net = ie_core.load_network(net, device)
    img = read_image()
    request = exec_net.requests[0]
    request.infer({'data': img})
    pc = request.get_perf_counts()
    assert pc['29']["status"] == "EXECUTED"
    assert pc['29']["layer_type"] == "FullyConnected"
    del exec_net
    del ie_core
    del net


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU", reason="Can't run test on device {},"
                                                                   "Dynamic batch fully supported only on CPU".format(os.environ.get("TEST_DEVICE", "CPU")))
def test_set_batch_size(device):
    xml = os.path.join(os.path.dirname(__file__), 'test_data', 'models', 'SampLeNet.xml')
    bin = os.path.join(os.path.dirname(__file__), 'test_data', 'models', 'SampLeNet.bin')
    ie_core = ie.IECore()
    ie_core.set_config({"DYN_BATCH_ENABLED": "YES"}, device)
    net = ie_core.read_network(xml, bin)
    net.batch_size = 10
    data = np.zeros(shape=net.inputs['data'].shape)
    exec_net = ie_core.load_network(net, device)
    data[0] = read_image()[0]
    request = exec_net.requests[0]
    request.set_batch(1)
    request.infer({'data': data})
    assert np.allclose(int(request.outputs['fc3'][0][0]), -1), "Incorrect data for 1st batch"
    del exec_net
    del ie_core
    del net


def test_set_zero_batch_size(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    exec_net = ie_core.load_network(net, device, num_requests=1)
    request = exec_net.requests[0]
    with pytest.raises(ValueError) as e:
        request.set_batch(0)
    assert "Batch size should be positive integer number but 0 specified" in str(e.value)
    del exec_net
    del ie_core
    del net


def test_set_negative_batch_size(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    exec_net =  ie_core.load_network(net, device, num_requests=1)
    request = exec_net.requests[0]
    with pytest.raises(ValueError) as e:
        request.set_batch(-1)
    assert "Batch size should be positive integer number but -1 specified" in str(e.value)
    del exec_net
    del ie_core
    del net
