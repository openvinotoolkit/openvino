import warnings
import numpy

from openvino.inference_engine import DataPtr, IECore
from conftest import model_path


test_net_xml, test_net_bin = model_path()

def test_name():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert net.layers['27'].name == "27"


def test_type():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert net.layers['27'].type == "Pooling"


def test_precision_getter(recwarn):
    warnings.simplefilter("always")
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert net.layers['27'].precision == "FP32"
    assert len(recwarn) == 1
    assert recwarn.pop(DeprecationWarning)

def test_precision_setter(recwarn):
    warnings.simplefilter("always")
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.layers['27'].precision = "I8"
    assert net.layers['27'].precision == "I8"
    assert len(recwarn) == 1
    assert recwarn.pop(DeprecationWarning)

def test_affinuty_getter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert net.layers['27'].affinity == ""


def test_affinity_setter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.layers['27'].affinity = "CPU"
    assert net.layers['27'].affinity == "CPU"


def test_blobs():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert isinstance(net.layers['19/Fused_Add_'].blobs["biases"], numpy.ndarray)
    assert isinstance(net.layers['19/Fused_Add_'].blobs["weights"], numpy.ndarray)
    assert net.layers['19/Fused_Add_'].blobs["biases"].size != 0
    assert net.layers['19/Fused_Add_'].blobs["weights"].size != 0

def test_weights(recwarn):
    warnings.simplefilter("always")
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert isinstance(net.layers['19/Fused_Add_'].weights["biases"], numpy.ndarray)
    assert isinstance(net.layers['19/Fused_Add_'].weights["weights"], numpy.ndarray)
    assert net.layers['19/Fused_Add_'].weights["biases"].size != 0
    assert net.layers['19/Fused_Add_'].weights["weights"].size != 0
    assert len(recwarn) == 4
    assert recwarn.pop(DeprecationWarning)


def test_params_getter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert net.layers['27'].params == {"kernel" : "2,2", "pads_begin" : "0,0",
                                       "pads_end" : "0,0", "rounding_type" : "floor",
                                       "strides" : "2,2", "pool-method" : "max",
                                       "originalLayersNames" : "27"}


def test_params_setter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    params = net.layers['27'].params
    params.update({'PrimitivesPriority': 'cpu:ref_any'})
    net.layers['27'].params = params
    assert net.layers['27'].params == {"kernel" : "2,2", "pads_begin" : "0,0",
                                       "pads_end" : "0,0", "rounding_type" : "floor",
                                       "strides" : "2,2", "pool-method" : "max",
                                       "originalLayersNames" : "27", 'PrimitivesPriority': 'cpu:ref_any'}


def test_layer_parents():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert net.layers['27'].parents == ['26']


def test_layer_children():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert net.layers['27'].children == ['29']


def test_layout(recwarn):
    warnings.simplefilter("always")
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert net.layers['27'].layout == 'NCHW'
    assert len(recwarn) == 1
    assert recwarn.pop(DeprecationWarning)


def test_shape(recwarn):
    warnings.simplefilter("always")
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert net.layers['27'].shape == [1, 64, 9, 9]
    assert len(recwarn) == 1


def test_out_data():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert isinstance(net.layers['27'].out_data[0], DataPtr)

def test_in_data():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert isinstance(net.layers['27'].in_data[0], DataPtr)
