import warnings
import os
import numpy

from openvino.inference_engine import DataPtr, IECore

SAMPLENET_XML = os.path.join(os.path.dirname(__file__), 'test_data', 'models', 'test_model.xml')
SAMPLENET_BIN = os.path.join(os.path.dirname(__file__), 'test_data', 'models', 'test_model.bin')


def test_name():
    ie = IECore()
    net = ie.read_network(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    assert net.layers['19'].name == "19"


def test_type():
    ie = IECore()
    net = ie.read_network(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    assert net.layers['19'].type == "Convolution"


def test_precision_getter(recwarn):
    warnings.simplefilter("always")
    ie = IECore()
    net = ie.read_network(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    assert net.layers['19'].precision == "FP32"
    assert len(recwarn) == 1
    assert recwarn.pop(DeprecationWarning)

def test_precision_setter(recwarn):
    warnings.simplefilter("always")
    ie = IECore()
    net = ie.read_network(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    net.layers['19'].precision = "I8"
    assert net.layers['19'].precision == "I8"
    assert len(recwarn) == 1
    assert recwarn.pop(DeprecationWarning)

def test_affinuty_getter():
    ie = IECore()
    net = ie.read_network(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    assert net.layers['19'].affinity == ""


def test_affinity_setter():
    ie = IECore()
    net = ie.read_network(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    net.layers['19'].affinity = "CPU"
    assert net.layers['19'].affinity == "CPU"


def test_blobs():
    ie = IECore()
    net = ie.read_network(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    assert isinstance(net.layers['19'].blobs["biases"], numpy.ndarray)
    assert isinstance(net.layers['19'].blobs["weights"], numpy.ndarray)
    assert net.layers['19'].blobs["biases"].size != 0
    assert net.layers['19'].blobs["weights"].size != 0

def test_weights(recwarn):
    warnings.simplefilter("always")
    ie = IECore()
    net = ie.read_network(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    assert isinstance(net.layers['19'].weights["biases"], numpy.ndarray)
    assert isinstance(net.layers['19'].weights["weights"], numpy.ndarray)
    assert net.layers['19'].weights["biases"].size != 0
    assert net.layers['19'].weights["weights"].size != 0
    assert len(recwarn) == 4
    assert recwarn.pop(DeprecationWarning)


def test_params_getter():
    ie = IECore()
    net = ie.read_network(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    assert net.layers['19'].params == {'dilations': '1,1', 'group': '1', 'kernel': '5,5', 'output': '16', 'pads_begin': '2,2',
                              'pads_end': '2,2', 'strides': '1,1'}


def test_params_setter():
    ie = IECore()
    net = ie.read_network(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    params = net.layers['19'].params
    params.update({'PrimitivesPriority': 'cpu:ref_any'})
    net.layers['19'].params = params
    assert net.layers['19'].params == {'dilations': '1,1', 'group': '1', 'kernel': '5,5', 'output': '16',
                                       'pads_begin': '2,2',
                                       'pads_end': '2,2', 'strides': '1,1', 'PrimitivesPriority': 'cpu:ref_any'}


def test_layer_parents():
    ie = IECore()
    net = ie.read_network(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    assert net.layers['19'].parents == ['data']


def test_layer_children():
    ie = IECore()
    net = ie.read_network(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    assert net.layers['19'].children == ['21']


def test_layout(recwarn):
    warnings.simplefilter("always")
    ie = IECore()
    net = ie.read_network(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    assert net.layers['19'].layout == 'NCHW'
    assert len(recwarn) == 1
    assert recwarn.pop(DeprecationWarning)


def test_shape(recwarn):
    warnings.simplefilter("always")
    ie = IECore()
    net = ie.read_network(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    assert net.layers['19'].shape == [1, 16, 32, 32]
    assert len(recwarn) == 1


def test_out_data():
    ie = IECore()
    net = ie.read_network(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    assert isinstance(net.layers['19'].out_data[0], DataPtr)

def test_in_data():
    ie = IECore()
    net = ie.read_network(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    assert isinstance(net.layers['19'].in_data[0], DataPtr)
