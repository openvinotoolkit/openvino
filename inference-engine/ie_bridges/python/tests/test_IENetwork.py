import os
import pytest
import warnings
import numpy as np

from openvino.inference_engine import IECore, IENetwork, IENetLayer, DataPtr, \
    InputInfoPtr, PreProcessInfo
from conftest import model_path


test_net_xml, test_net_bin = model_path()


def test_create_ie_network_deprecated():
    with warnings.catch_warnings(record=True) as w:
        net = IENetwork(model=test_net_xml, weights=test_net_bin)
        assert isinstance(net, IENetwork)
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "Reading network using constructor is deprecated. " \
               "Please, use IECore.read_network() method instead" in str(w[0].message)


def test_incorrect_xml_deprecated():
    with warnings.catch_warnings(record=True) as w:
        with pytest.raises(Exception) as e:
            IENetwork(model="./model.xml", weights=test_net_bin)
        assert "Path to the model ./model.xml doesn't exist or it's a directory" in str(e.value)
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "Reading network using constructor is deprecated. " \
               "Please, use IECore.read_network() method instead" in str(w[0].message)


def test_incorrect_bin_deprecated():
    with warnings.catch_warnings(record=True) as w:
        with pytest.raises(Exception) as e:
            IENetwork(model=test_net_xml, weights="./model.bin")
        assert "Path to the weights ./model.bin doesn't exist or it's a directory" in str(e.value)
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "Reading network using constructor is deprecated. " \
               "Please, use IECore.read_network() method instead" in str(w[0].message)


def test_name():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert net.name == "test_model"


def test_inputs_deprecated():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    with warnings.catch_warnings(record=True) as w:
        inp = net.inputs
        assert isinstance(inp['data'], DataPtr)
        assert inp['data'].layout == "NCHW"
        assert inp['data'].precision == "FP32"
        assert inp['data'].shape == [1, 3, 32, 32]
    assert len(w) == 1
    assert "'inputs' property of IENetwork class is deprecated. " \
               "To access DataPtrs user need to use 'input_data' property " \
               "of InputInfoPtr objects which " \
               "can be accessed by 'input_info' property." in str(w[-1].message)


def test_input_info():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert isinstance(net.input_info['data'], InputInfoPtr)
    assert net.input_info['data'].layout == "NCHW"
    assert net.input_info['data'].precision == "FP32"
    assert isinstance(net.input_info['data'].input_data, DataPtr)
    assert isinstance(net.input_info['data'].preprocess_info, PreProcessInfo)


def test_input_info_precision_setter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert net.input_info['data'].layout == "NCHW"
    net.input_info['data'].layout = "NHWC"
    assert net.input_info['data'].layout == "NHWC"


def test_input_input_info_layout_setter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert net.input_info['data'].precision == "FP32"
    net.input_info['data'].precision = "I8"
    assert net.input_info['data'].precision == "I8"


def test_input_unsupported_precision_setter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    with pytest.raises(ValueError) as e:
        net.input_info['data'].precision = "BLA"
    assert "Unsupported precision BLA! List of supported precisions: " in str(e.value)


def test_input_unsupported_layout_setter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    with pytest.raises(ValueError) as e:
        net.input_info['data'].layout = "BLA"
    assert "Unsupported layout BLA! List of supported layouts: " in str(e.value)


def test_outputs():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert isinstance(net.outputs['fc_out'], DataPtr)
    assert net.outputs['fc_out'].layout == "NC"
    assert net.outputs['fc_out'].precision == "FP32"
    assert net.outputs['fc_out'].shape == [1, 10]


def test_output_precision_setter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert net.outputs['fc_out'].precision == "FP32"
    net.outputs['fc_out'].precision = "I8"
    assert net.outputs['fc_out'].precision == "I8"


def test_output_unsupported_precision_setter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    with pytest.raises(ValueError) as e:
        net.outputs['fc_out'].precision = "BLA"
    assert "Unsupported precision BLA! List of supported precisions: " in str(e.value)


def test_add_ouputs():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.add_outputs('28/Reshape')
    net.add_outputs(['29/WithoutBiases'])
    assert sorted(net.outputs) == ['28/Reshape', '29/WithoutBiases', 'fc_out']


def test_add_outputs_with_port():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.add_outputs(('28/Reshape', 0))
    net.add_outputs([('29/WithoutBiases', 0)])
    assert sorted(net.outputs) == ['28/Reshape', '29/WithoutBiases', 'fc_out']


def test_add_outputs_with_and_without_port():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.add_outputs('28/Reshape')
    net.add_outputs([('29/WithoutBiases', 0)])
    assert sorted(net.outputs) == ['28/Reshape', '29/WithoutBiases', 'fc_out']


def test_batch_size_getter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert net.batch_size == 1


def test_batch_size_setter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.batch_size = 4
    assert net.batch_size == 4
    assert net.input_info['data'].input_data.shape == [4, 3, 32, 32]


def test_batch_size_after_reshape():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.reshape({'data': [4, 3, 32, 32]})
    assert net.batch_size == 4
    assert net.input_info['data'].input_data.shape == [4, 3, 32, 32]
    net.reshape({'data': [8, 3, 32, 32]})
    assert net.batch_size == 8
    assert net.input_info['data'].input_data.shape == [8, 3, 32, 32]


def test_layers():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    layers_name = [key for key in net.layers]
    assert sorted(layers_name) == ['19/Fused_Add_', '21', '22', '23', '24/Fused_Add_',
                                   '26', '27', '29', 'data', 'fc_out']
    assert isinstance(net.layers['19/Fused_Add_'], IENetLayer)


@pytest.mark.skip(reason="Test is failed due-to ngraph conversion")
def test_serialize():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.serialize("./serialized_net.xml", "./serialized_net.bin")
    serialized_net = ie.read_network(model="./serialized_net.xml", weights="./serialized_net.bin")
    assert net.layers.keys() == serialized_net.layers.keys()
    os.remove("./serialized_net.xml")
    os.remove("./serialized_net.bin")


def test_reshape():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.reshape({"data": (2, 3, 32, 32)})


def test_read_net_from_buffer_deprecated():
    with warnings.catch_warnings(record=True) as w:
        with open(test_net_bin, 'rb') as f:
            bin = f.read()
        with open(test_net_xml, 'rb') as f:
            xml = f.read()
        net = IENetwork(model=xml, weights=bin, init_from_buffer=True)
        assert isinstance(net, IENetwork)
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "Reading network using constructor is deprecated. " \
               "Please, use IECore.read_network() method instead" in str(w[0].message)


def test_net_from_buffer_valid_deprecated():
    with warnings.catch_warnings(record=True) as w:
        with open(test_net_bin, 'rb') as f:
            bin = f.read()
        with open(test_net_xml, 'rb') as f:
            xml = f.read()
        net = IENetwork(model=xml, weights=bin, init_from_buffer=True)
        net2 = IENetwork(model=test_net_xml, weights=test_net_bin)
        for name, l in net.layers.items():
            for blob, data in l.blobs.items():
                assert np.allclose(data, net2.layers[name].blobs[blob]), \
                    "Incorrect weights for layer {} and blob {}".format(name, blob)
        assert len(w) == 2
        for warns in w:
            assert issubclass(warns.category, DeprecationWarning)
            assert "Reading network using constructor is deprecated. " \
                   "Please, use IECore.read_network() method instead" in str(warns.message)


def test_multi_out_data():
    # Regression test CVS-23965
    # Check that DataPtr for all output layers not copied between outputs map  items
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.add_outputs(['28/Reshape'])
    assert "28/Reshape" in net.outputs and "fc_out" in net.outputs
    assert isinstance(net.outputs["28/Reshape"], DataPtr)
    assert isinstance(net.outputs["fc_out"], DataPtr)
    assert net.outputs["28/Reshape"].name == "28/Reshape" and net.outputs["28/Reshape"].shape == [1, 5184]
    assert net.outputs["fc_out"].name == "fc_out" and net.outputs["fc_out"].shape == [1, 10]
    pass
